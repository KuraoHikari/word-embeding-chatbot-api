import { verify } from "hono/jwt";

import env from "@/env";
import { createRouter } from "@/lib/create-app";
import { authMiddleware } from "@/middlewares/auth.middleware";
import { bodyLimitMiddleware } from "@/middlewares/body-limit.middleware";
import { chatbotsLimiter } from "@/middlewares/limiter.middleware";

import * as handlers from "./chatbots.handler";
import * as routes from "./chatbots.routes";

const router = createRouter(); // Apply auth middleware to all routes
router.use(routes.list.path, authMiddleware);
router.use(routes.create.path, authMiddleware, bodyLimitMiddleware(12 * 1024 * 1024), chatbotsLimiter);
// Allow GET /chatbots/:id without auth (for widget public access)
// Require auth for PATCH, DELETE
router.use("/chatbots/:id", async (c, next) => {
  if (c.req.method === "GET") {
    const authHeader = c.req.header("Authorization");
    if (authHeader?.startsWith("Bearer ")) {
      try {
        const token = authHeader.slice(7);
        const payload = await verify(token, env.ACCESS_TOKEN_SECRET);
    const userId = typeof payload.sub === "string" ? Number(payload.sub) : payload.sub;
    if (typeof userId === "number" && Number.isFinite(userId)) {
      c.set("userId", userId);
    }
      }
      catch {
        // Token invalid — proceed as unauthenticated public access
      }
    }
    return next();
  }
  return authMiddleware(c, next);
});

// Then register OpenAPI routes
router.openapi(routes.list, handlers.list);
router.openapi(routes.create, handlers.create);
router.openapi(routes.getOne, handlers.getOne);
router.openapi(routes.patch, handlers.patch);
router.openapi(routes.remove, handlers.remove);

export default router;
