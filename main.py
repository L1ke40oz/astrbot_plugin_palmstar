import hmac
import time

from aiohttp import web

from astrbot.api import AstrBotConfig, logger
from astrbot.api.message_components import Plain
from astrbot.api.platform import AstrBotMessage, MessageMember, MessageType
from astrbot.api.star import Context, Star

DEFAULT_PART_DESCRIPTIONS = """head=（她轻轻摸了摸你的头）
hand=（她牵起了你的手）
face=（她用指尖碰了碰你的脸）"""


class PalmStarPlugin(Star):
    """Turns a BJD doll with an ESP32 into the bot's body.

    The ESP32 sends ``GET /touch?part=head&token=...`` when touched. The plugin
    converts that into an event description and injects it into the owner's QQ
    private session as if the owner had sent it, so AstrBot replies through the
    normal pipeline (persona, memory, conversation context).
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

    async def initialize(self):
        app = web.Application()
        # Future phases add routes here, e.g. a /line endpoint for ESP32
        # reporting a played voice line id.
        app.router.add_get("/touch", self.handle_touch)
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
            logger.warning("[palmstar] token is empty, /touch accepts any request")

    async def terminate(self):
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            logger.info("[palmstar] HTTP server stopped")

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

        if not self.target_qq.isdigit():
            logger.error("[palmstar] target_qq is not configured")
            return web.json_response(
                {"ok": False, "error": "target_qq not configured"}, status=503
            )

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
            return web.json_response(
                {"ok": False, "error": "platform not ready"}, status=503
            )

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

        text = description
        if pressure is not None and self.pressure_template:
            text += self.pressure_template.format(pressure=pressure)

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

        event = platform.create_event(abm)
        # Marker for future phases (e.g. turning the reply into speech for the
        # ESP32): hooks can check event.get_extra("palmstar_touch").
        event.set_extra("palmstar_touch", {"part": part, "pressure": pressure})
        platform.commit_event(event)
        self._last_touch[part] = now

        logger.info(
            f"[palmstar] Injected touch part={part} pressure={pressure}: {text}"
        )
        return web.json_response({"ok": True, "text": text})
