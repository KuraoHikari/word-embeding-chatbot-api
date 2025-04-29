import configureOpenAPI from "@/lib/configure-open-api";
import createApp from "@/lib/create-app";
import auth from "@/routes/auth/auth.index";
import chatbot from "@/routes/chatbots/chatbots.index";
import index from "@/routes/index.route";
import tasks from "@/routes/tasks/tasks.index";

const app = createApp();

const routes = [index, tasks, auth, chatbot]; // Add your routes here

configureOpenAPI(app);
routes.forEach((route) => {
  app.route("/", route);
});

export default app;
