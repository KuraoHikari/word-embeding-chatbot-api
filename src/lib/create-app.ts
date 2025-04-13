import { OpenAPIHono } from "@hono/zod-openapi";
import { cors } from "hono/cors";
import { notFound, onError } from "stoker/middlewares";
import { defaultHook } from "stoker/openapi";

import { apiLimiter } from "@/middlewares/limiter.middleware";
import { pinoLogger } from "@/middlewares/pino-logger";

import type { AppBindings } from "./types";

export function createRouter() {
  return new OpenAPIHono<AppBindings>({
    strict: false,
    defaultHook,
  });
}

export default function createApp() {
  const app = createRouter();
  app.use("*", cors({ origin: "*" })); // Allow CORS for all origins
  app.use("*", apiLimiter);
  app.use(pinoLogger());

  app.notFound(notFound);
  app.onError(onError);
  return app;
}
