import { createRouter } from "@/lib/create-app";
import { authMiddleware } from "@/middlewares/auth.middleware";

import * as handlers from "./contacts.handler";
import * as routes from "./contacts.routes";

const router = createRouter(); // Apply auth middleware to all routes

// router.use(routes.list.path, authMiddleware);

router.openapi(routes.list, handlers.list);
router.openapi(routes.create, handlers.create);
// router.openapi(routes.getOne, handlers.getOne);
router.openapi(routes.remove, handlers.remove);

export default router;
