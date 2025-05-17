import { createRouter } from "@/lib/create-app";
import { authMiddleware, authMiddlewarePublicContact } from "@/middlewares/auth.middleware";

import * as handlers from "./messages.handler";
import * as routes from "./messages.routes";

const router = createRouter(); // Apply auth middleware to all routes

router.use(routes.create.path, authMiddlewarePublicContact);

router.openapi(routes.create, handlers.create);

export default router;
