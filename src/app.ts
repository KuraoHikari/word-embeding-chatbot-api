import configureOpenAPI from "@/lib/configure-open-api";
import createApp from "@/lib/create-app";
import auth from "@/routes/auth/auth.index";
import chatbot from "@/routes/chatbots/chatbots.index";
import contacts from "@/routes/contacts/contacts.index";
import conversations from "@/routes/conversations/conversations.index";
import index from "@/routes/index.route";
import messages from "@/routes/messages/messages.index";
import tasks from "@/routes/tasks/tasks.index";

const app = createApp();

const routes = [index, tasks, auth, chatbot, contacts, conversations, messages]; // Add your routes here

configureOpenAPI(app);
routes.forEach((route) => {
  app.route("/", route);
});

export default app;
