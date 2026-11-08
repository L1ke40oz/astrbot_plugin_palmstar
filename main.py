import array
import asyncio
import hmac
import re
import time
import wave
from pathlib import Path

from aiohttp import web

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.platform import AstrBotMessage, MessageMember, MessageType
from astrbot.api.provider import STTProvider
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.provider.provider import TTSProvider
from astrbot.core.star import star_map
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

DEFAULT_PART_DESCRIPTIONS = """head=（她轻轻摸了摸你的头）
hand=（她牵起了你的手）
face=（她用指尖碰了碰你的脸）"""
DEFAULT_VOICE_TEMPLATE = "（她握着你的手，对你说）{text}"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # ~5 minutes of 16 kHz 16-bit mono PCM
SPEECH_RATE = 16000  # Sample rate of the PCM sent to the ESP32 speaker
# Stage directions like （笑） or *pats your head* should not be read aloud.
BRACKET_PATTERN = re.compile(r"（[^）]*）|\([^)]*\)|\*[^*]*\*")


class PalmStarPlugin(Star):
    """Turns a BJD doll with an ESP32 into the bot's body.

    The ESP32 reports touches (``GET /touch``) and recorded speech
    (``POST /voice``). The plugin turns them into text and injects it into the
    owner's QQ private session as if the owner had sent it, so AstrBot replies
    through the normal pipeline (persona, memory, conversation context). The
    reply is also synthesized to speech, which the ESP32 fetches from
    ``GET /speech`` and plays.
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.port = int(config.get("port", 8765))
        self.token = str(config.get("token", "")).strip()
        self.target_qq = str(config.get("target_qq", "")).strip()
        self.platform_id = str(config.get("platform_id", "")).strip()
        self.cooldown = float(config.get("cooldown_seconds", 10))
        self.pressure_template = str(config.get("pressure_template", ""))
        self.stt_provider_id = str(config.get("stt_provider_id", "")).strip()
        self.save_voice_debug = bool(config.get("save_voice_debug", False))
        self.tts_provider_id = str(config.get("tts_provider_id", "")).strip()
        self.speech_strip_brackets = bool(config.get("speech_strip_brackets", True))
        self.voice_template = (
            str(config.get("voice_template", "")).strip() or DEFAULT_VOICE_TEMPLATE
        )
        self.bypass_plugins = {
            str(name).strip() for name in config.get("bypass_plugins", []) if name
        }

        # "head=（她轻轻摸了摸你的头）" per line -> {"head": "（她轻轻摸了摸你的头）"}
        self.part_descriptions: dict[str, str] = {}
        raw = config.get("part_descriptions") or DEFAULT_PART_DESCRIPTIONS
        for line in str(raw).splitlines():
            key, sep, text = line.partition("=")
            if sep and key.strip() and text.strip():
                self.part_descriptions[key.strip().lower()] = text.strip()

        self._last_touch: dict[str, float] = {}  # part -> monotonic timestamp
        self._self_id = ""
        self._nickname = ""
        self._runner: web.AppRunner | None = None
        # Injected event id -> future resolved with the reply as 16 kHz PCM
        # (b"" when there is nothing to play). Filled by speak_reply, consumed
        # by handle_speech.
        self._event_seq = 0
        self._speech: dict[int, asyncio.Future] = {}

    async def initialize(self):
        app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
        app.router.add_get("/touch", self.handle_touch)
        app.router.add_post("/voice", self.handle_voice)
        app.router.add_get("/speech", self.handle_speech)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        try:
            await web.TCPSite(self._runner, "0.0.0.0", self.port).start()
        except OSError as e:
            logger.error(f"[palmstar] Failed to listen on port {self.port}: {e}")
            await self._runner.cleanup()
            self._runner = None
            return
        logger.info(
            f"[palmstar] HTTP server listening on 0.0.0.0:{self.port}, "
            f"parts={list(self.part_descriptions)}"
        )
        if not self.token:
            logger.warning("[palmstar] token is empty, requests are not checked")

    async def terminate(self):
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            logger.info("[palmstar] HTTP server stopped")

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=10000)
    async def skip_plugins_for_doll_events(self, event: AstrMessageEvent):
        """Keep injected doll events away from plugins listed in bypass_plugins.

        AstrBot runs message handlers from the ``activated_handlers`` list in
        priority order. This handler runs first (very high priority) and removes
        the listed plugins' handlers from that list, so e.g. a debounce plugin
        never holds a doll event back. Real user messages are untouched.

        Args:
            event: Incoming private message event.
        """
        if event.get_extra("palmstar") is None or not self.bypass_plugins:
            return
        handlers = event.get_extra("activated_handlers") or []
        # Mutate in place: the pipeline is iterating this very list, and every
        # removed handler sits after this one because of the priority above.
        for handler in list(handlers):
            md = star_map.get(handler.handler_module_path)
            if md and md.name in self.bypass_plugins:
                handlers.remove(handler)
                logger.info(f"[palmstar] Skipped {md.name} for doll event")

    async def handle_touch(self, request: web.Request) -> web.Response:
        """Handle a touch report from the ESP32.

        Query args:
            part: Touched body part, must be a key in part_descriptions.
            token: Shared secret, must match the configured token if set.
            pressure: Optional pressure value, integer 0-100.

        Returns:
            JSON response. 200 injected, 400 bad args, 403 bad token,
            429 cooling down, 503 platform or target not ready.
        """
        q = request.query
        if self.token and not hmac.compare_digest(
            q.get("token", "").encode(), self.token.encode()
        ):
            return web.json_response({"ok": False, "error": "bad token"}, status=403)

        part = q.get("part", "").strip().lower()
        description = self.part_descriptions.get(part)
        if not description:
            return web.json_response(
                {"ok": False, "error": f"unknown part: {part!r}"}, status=400
            )

        pressure = None
        if q.get("pressure"):
            try:
                pressure = int(q["pressure"])
            except ValueError:
                pressure = -1
            if not 0 <= pressure <= 100:
                return web.json_response(
                    {"ok": False, "error": "pressure must be 0-100"}, status=400
                )

        now = time.monotonic()
        remaining = self.cooldown - (now - self._last_touch.get(part, -1e9))
        if remaining > 0:
            return web.json_response(
                {"ok": False, "error": "cooldown", "retry_after": round(remaining, 1)},
                status=429,
            )

        text = description
        if pressure is not None and self.pressure_template:
            text += self.pressure_template.format(pressure=pressure)

        marker = {"type": "touch", "part": part, "pressure": pressure}
        error = await self._inject(text, marker)
        if error:
            return web.json_response({"ok": False, "error": error}, status=503)
        self._last_touch[part] = now
        logger.info(
            f"[palmstar] Injected touch part={part} pressure={pressure}: {text}"
        )
        return web.json_response({"ok": True, "text": text, "id": marker["id"]})

    async def handle_voice(self, request: web.Request) -> web.Response:
        """Handle speech recorded by the ESP32 while the hand FSR is pressed.

        The request body is raw 16-bit little-endian mono PCM. It is saved as a
        WAV file, transcribed by an AstrBot speech-to-text provider, wrapped in
        voice_template and injected like a touch event.

        Query args:
            token: Shared secret, must match the configured token if set.
            rate: Sample rate of the PCM body, default 16000.

        Returns:
            JSON response. 200 injected, 400 bad body, 403 bad token,
            422 no speech recognized, 502 transcription failed,
            503 STT provider, platform or target not ready.
        """
        q = request.query
        if self.token and not hmac.compare_digest(
            q.get("token", "").encode(), self.token.encode()
        ):
            return web.json_response({"ok": False, "error": "bad token"}, status=403)

        try:
            rate = int(q.get("rate", 16000))
        except ValueError:
            rate = 0
        if not 8000 <= rate <= 48000:
            return web.json_response(
                {"ok": False, "error": "rate must be 8000-48000"}, status=400
            )
        pcm = await request.read()
        seconds = len(pcm) / (rate * 2)
        # Level stats make mic problems visible in the log: peak near 32767 with
        # a high clipped share means distortion, a tiny peak means near silence.
        samples = array.array("h", pcm[: len(pcm) // 2 * 2])
        peak = max(map(abs, samples), default=0)
        clipped = sum(1 for s in samples if abs(s) >= 32000) / max(len(samples), 1)
        logger.info(
            f"[palmstar] Voice upload {seconds:.1f}s rate={rate} "
            f"peak={peak} clipped={clipped:.1%}"
        )
        if seconds < 0.3:
            return web.json_response(
                {"ok": False, "error": "audio too short"}, status=400
            )

        # A configured provider id lets the doll use STT without turning on
        # AstrBot's global speech-to-text switch.
        if self.stt_provider_id:
            provider = self.context.get_provider_by_id(self.stt_provider_id)
        else:
            provider = self.context.get_using_stt_provider()
        if not isinstance(provider, STTProvider):
            logger.error(
                f"[palmstar] STT provider not available (id={self.stt_provider_id!r})"
            )
            return web.json_response(
                {"ok": False, "error": "stt provider not ready"}, status=503
            )

        # Debug mode keeps every upload in the plugin data dir so it can be
        # listened to; otherwise the WAV is a temp file deleted after use.
        wav_dir = (
            StarTools.get_data_dir("astrbot_plugin_palmstar") / "voice_debug"
            if self.save_voice_debug
            else Path(get_astrbot_temp_path())
        )
        wav_path = wav_dir / f"palmstar_voice_{int(time.time() * 1000)}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm[: len(pcm) // 2 * 2])
            heard = (await provider.get_text(str(wav_path)) or "").strip()
        except Exception as e:
            logger.error(f"[palmstar] Speech transcription failed: {e}")
            return web.json_response(
                {"ok": False, "error": "transcription failed"}, status=502
            )
        finally:
            if self.save_voice_debug:
                logger.info(f"[palmstar] Saved voice upload to {wav_path}")
            else:
                wav_path.unlink(missing_ok=True)

        if not heard:
            logger.info(f"[palmstar] No speech recognized in {seconds:.1f}s audio")
            return web.json_response(
                {"ok": False, "error": "no speech recognized"}, status=422
            )

        if "{text}" in self.voice_template:
            text = self.voice_template.replace("{text}", heard)
        else:
            text = self.voice_template + heard
        marker = {"type": "voice", "seconds": seconds}
        error = await self._inject(text, marker)
        if error:
            return web.json_response({"ok": False, "error": error}, status=503)
        logger.info(f"[palmstar] Injected voice ({seconds:.1f}s): {text}")
        return web.json_response({"ok": True, "text": text, "id": marker["id"]})

    @filter.after_message_sent()
    async def speak_reply(self, event: AstrMessageEvent):
        """Turn the bot's reply to a doll event into speech for the ESP32.

        Runs after the reply was sent to QQ, takes its text, synthesizes it with
        an AstrBot TTS provider and converts it to 16 kHz mono PCM with ffmpeg.
        The result resolves the future /speech is waiting on.

        Args:
            event: The event whose reply was just sent.
        """
        marker = event.get_extra("palmstar")
        future = self._speech.get(marker.get("id")) if marker else None
        if future is None or future.done():
            return

        result = event.get_result()
        text = "".join(
            c.text for c in (result.chain if result else []) if isinstance(c, Plain)
        )
        if self.speech_strip_brackets:
            text = BRACKET_PATTERN.sub("", text)
        text = text.strip()

        # A configured provider id lets the doll speak without turning on
        # AstrBot's global TTS switch, so QQ replies stay text.
        if self.tts_provider_id:
            provider = self.context.get_provider_by_id(self.tts_provider_id)
        else:
            provider = self.context.get_using_tts_provider()
        pcm = b""
        if not text:
            logger.info("[palmstar] Reply has no speakable text")
        elif not isinstance(provider, TTSProvider):
            logger.warning(
                f"[palmstar] TTS provider not available (id={self.tts_provider_id!r})"
            )
        else:
            try:
                audio_path = await provider.get_audio(text)
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-v", "error", "-i", str(audio_path),
                    "-ac", "1", "-ar", str(SPEECH_RATE), "-f", "s16le", "-",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )  # fmt: skip
                pcm, err = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(err.decode(errors="ignore").strip()[-300:])
                logger.info(
                    f"[palmstar] Speech ready ({len(pcm) / (SPEECH_RATE * 2):.1f}s): "
                    f"{text}"
                )
            except Exception as e:
                logger.error(f"[palmstar] Text-to-speech failed: {e}")
                pcm = b""
        if not future.done():
            future.set_result(pcm)

    async def handle_speech(self, request: web.Request) -> web.Response:
        """Let the ESP32 wait for and download the spoken reply to an event.

        Query args:
            id: Event id returned by /touch or /voice.
            token: Shared secret, must match the configured token if set.
            wait: Seconds to wait for the reply, default 60, max 120.

        Returns:
            200 with raw 16 kHz 16-bit mono PCM, 204 if there is nothing to play
            (no reply, no TTS, timeout), 400 bad args, 403 bad token,
            404 unknown or already fetched id.
        """
        q = request.query
        if self.token and not hmac.compare_digest(
            q.get("token", "").encode(), self.token.encode()
        ):
            return web.json_response({"ok": False, "error": "bad token"}, status=403)
        try:
            event_id = int(q["id"])
            wait = min(float(q.get("wait", 60)), 120)
        except (KeyError, ValueError):
            return web.json_response(
                {"ok": False, "error": "id and wait must be numbers"}, status=400
            )
        future = self._speech.get(event_id)
        if future is None:
            return web.json_response({"ok": False, "error": "unknown id"}, status=404)
        try:
            pcm = await asyncio.wait_for(asyncio.shield(future), wait)
        except asyncio.TimeoutError:
            return web.Response(status=204)
        self._speech.pop(event_id, None)
        if not pcm:
            return web.Response(status=204)
        return web.Response(
            body=pcm,
            content_type="application/octet-stream",
            headers={"X-Sample-Rate": str(SPEECH_RATE)},
        )

    async def _inject(self, text: str, marker: dict) -> str | None:
        """Inject text into the owner's QQ private chat as if the owner sent it.

        Args:
            text: Message text the bot will see.
            marker: Stored as event extra "palmstar" so hooks can recognize doll
                events. On success an "id" key is added; the ESP32 passes it to
                /speech to fetch the spoken reply.

        Returns:
            None on success, otherwise a short error string.
        """
        if not self.target_qq.isdigit():
            logger.error("[palmstar] target_qq is not configured")
            return "target_qq not configured"

        # Find the OneBot (aiocqhttp) adapter the owner chats through.
        platform = None
        for p in self.context.platform_manager.platform_insts:
            meta = p.meta()
            if (self.platform_id and meta.id == self.platform_id) or (
                not self.platform_id and meta.name == "aiocqhttp"
            ):
                platform = p
                break
        if platform is None:
            logger.error(
                f"[palmstar] platform not found (platform_id={self.platform_id!r})"
            )
            return "platform not ready"

        # Bot QQ id and owner nickname, fetched once from OneBot and cached.
        if not self._self_id:
            try:
                bot = platform.get_client()
                info = await bot.get_login_info()
                self._self_id = str(info.get("user_id", ""))
                stranger = await bot.get_stranger_info(user_id=int(self.target_qq))
                self._nickname = str(stranger.get("nickname", ""))
            except Exception as e:
                logger.warning(f"[palmstar] Failed to query OneBot info: {e}")

        # Same shape the aiocqhttp adapter builds for a real private message,
        # so the event goes through the full pipeline and replies to this chat.
        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.self_id = self._self_id
        abm.sender = MessageMember(self.target_qq, self._nickname or self.target_qq)
        abm.session_id = self.target_qq
        abm.message_id = f"palmstar_{int(time.time() * 1000)}"
        abm.message = [Plain(text)]
        abm.message_str = text
        abm.raw_message = None  # No OneBot event; sending falls back to session id.

        self._event_seq += 1
        marker["id"] = self._event_seq
        self._speech[marker["id"]] = asyncio.get_running_loop().create_future()
        while len(self._speech) > 20:  # Drop replies the ESP32 never fetched
            self._speech.pop(next(iter(self._speech)))

        event = platform.create_event(abm)
        event.set_extra("palmstar", marker)
        platform.commit_event(event)
        return None
