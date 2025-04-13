import { createRouter } from "@/lib/create-app";
import { authMiddleware } from "@/middlewares/auth.middleware";

import * as handlers from "./auth.handler";
import * as routes from "./auth.routes";
import { authLimiter } from "@/middlewares/limiter.middleware";

const router = createRouter();
router.use(routes.login.path, authLimiter); // Apply auth middleware to all routes
router.use(routes.getUser.path, authMiddleware);
router.use(routes.register.path, authLimiter);

// Then register OpenAPI routes
router.openapi(routes.login, handlers.login);
router.openapi(routes.register, handlers.register);
router.openapi(routes.getUser, handlers.getUser);

export default router;
