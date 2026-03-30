import type { TypedServer, TypedSocket } from "@/sockets/types";

import { setOffline, setOnline } from "@/services/presence.service";

export function registerPresenceHandlers(io: TypedServer, socket: TypedSocket): void {
 const userId = socket.data.userId;
 const contactId = socket.data.contactId;

 // ── Contact (widget) sockets — track contact presence ────────────
 if (contactId) {
  io.emit("presence:update", { userId, contactId, isOnline: true });

  socket.on("disconnect", () => {
   io.emit("presence:update", { userId, contactId, isOnline: false });
  });
  return;
 }

 // ── Admin sockets — track admin presence with Redis ──────────────
 (async () => {
  try {
   await setOnline(userId);
  } catch (err) {
   // Redis may be unreachable in dev — don't crash the socket
   console.error("[socket] presence online error (redis)", err);
  }
  // Always emit presence even if Redis fails so FE header updates
  io.emit("presence:update", { userId, isOnline: true });
 })();

 // ── On disconnect: mark offline ───────────────────────────────────
 socket.on("disconnect", async () => {
  try {
   await setOffline(userId);
  } catch (err) {
   console.error("[socket] presence offline error (redis)", err);
  }
  io.emit("presence:update", { userId, isOnline: false });
 });
}
