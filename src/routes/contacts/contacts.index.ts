import { createRouter } from "@/lib/create-app";
import { authMiddleware } from "@/middlewares/auth.middleware";

import * as handlers from "./contacts.handler";
import * as routes from "./contacts.routes";

const router = createRouter(); // Apply auth middleware to all routes

router.use("/contacts", async (c, next) => {
  if (c.req.method === "GET") {
    return authMiddleware(c, next);
  }

  return next();
});

router.use("/contacts/:id", async (c, next) => {
  if (c.req.method === "DELETE") {
    return authMiddleware(c, next);
  }

  return next();
});

router.openapi(routes.list, handlers.list);
router.openapi(routes.create, handlers.create);
// router.openapi(routes.getOne, handlers.getOne);
router.openapi(routes.remove, handlers.remove);

export default router;
