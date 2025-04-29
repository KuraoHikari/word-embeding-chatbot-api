import { createRouter } from "@/lib/create-app";
import { authMiddleware } from "@/middlewares/auth.middleware";

import * as handlers from "./chatbots.handler";
import * as routes from "./chatbots.routes";

const router = createRouter(); // Apply auth middleware to all routes
router.use(routes.list.path, authMiddleware);
router.use(routes.create.path, authMiddleware);

// Then register OpenAPI routes
router.openapi(routes.list, handlers.list);
router.openapi(routes.create, handlers.create);

export default router;
