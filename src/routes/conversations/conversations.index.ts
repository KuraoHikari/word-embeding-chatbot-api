import { createRouter } from "@/lib/create-app";
import { authMiddleware, authMiddlewarePublicContact } from "@/middlewares/auth.middleware";

import * as handlers from "./conversations.handler";
import * as routes from "./conversations.routes";

const router = createRouter(); // Apply auth middleware to all routes

router.use(routes.list.path, authMiddleware);
router.use(routes.remove.path, authMiddleware);

router.use(routes.create.path, authMiddlewarePublicContact);

router.openapi(routes.list, handlers.list);
router.openapi(routes.create, handlers.create);
router.openapi(routes.getOne, handlers.getOne);
router.openapi(routes.remove, handlers.remove);

export default router;
