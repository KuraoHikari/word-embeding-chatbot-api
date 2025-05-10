import { createRouter } from "@/lib/create-app";
import { authMiddleware } from "@/middlewares/auth.middleware";
import { bodyLimitMiddleware } from "@/middlewares/body-limit.middleware";
import { chatbotsLimiter } from "@/middlewares/limiter.middleware";

import * as handlers from "./chatbots.handler";
import * as routes from "./chatbots.routes";

const router = createRouter(); // Apply auth middleware to all routes
router.use(routes.list.path, authMiddleware);
router.use(routes.create.path, authMiddleware, bodyLimitMiddleware(12 * 1024 * 1024), chatbotsLimiter);
router.use("/chatbots/:id", authMiddleware);

// Then register OpenAPI routes
router.openapi(routes.list, handlers.list);
router.openapi(routes.create, handlers.create);
router.openapi(routes.patch, handlers.patch);
router.openapi(routes.remove, handlers.remove);

export default router;
