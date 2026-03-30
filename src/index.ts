import { getRequestListener } from "@hono/node-server";
import { createServer } from "node:http";

import app from "./app";
import env from "./env";
import { initSocketIO } from "./sockets";

const port = env.PORT;

// Create a raw Node HTTP server with Hono's request listener
const server = createServer(getRequestListener(app.fetch));

// Attach Socket.io to the same HTTP server
const _io = initSocketIO(server);

server.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Server is running on port http://localhost:${port}`);
});
