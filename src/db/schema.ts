import type { SQL } from "drizzle-orm";
import type { AnySQLiteColumn } from "drizzle-orm/sqlite-core";

import { relations, sql } from "drizzle-orm";
import { integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

const id = {
  id: integer("id", { mode: "number" })
    .primaryKey({ autoIncrement: true }),
};

const timestamps = {
  createdAt: integer("created_at", { mode: "timestamp" })
    .$defaultFn(() => new Date()),
  updatedAt: integer("updated_at", { mode: "timestamp" })
    .$defaultFn(() => new Date())
    .$onUpdate(() => new Date()),
};

export const tasks = sqliteTable("tasks", {
  ...id,
  name: text("name")
    .notNull(),
  done: integer("done", { mode: "boolean" })
    .notNull()
    .default(false),
  ...timestamps,
});

// custom lower function
export function lower(email: AnySQLiteColumn): SQL {
  return sql`lower(${email})`;
}

export const users = sqliteTable("users", {
  ...id,
  name: text("name").notNull(),
  password: text("password").notNull(),
  email: text("email").notNull(),
  ...timestamps,
}, table => ({
  emailUniqueIndex: uniqueIndex("emailUniqueIndex").on(lower(table.email)),
}));

// Chatbots Table
export const chatbots = sqliteTable("chatbots", {
  ...id,
  title: text("title").notNull(),
  description: text("description"),
  commandTemplate: text("command_template").notNull(),
  modelAi: text("model_ai").notNull(),
  embedingModel: text("embeding_model").notNull(),
  sugestionMessage: text("sugestion_message").notNull(),
  pdfTitle: text("pdf_title").notNull(),
  pdfLink: text("pdf_link").notNull(),
  userId: integer("user_id").notNull().references(() => users.id),
  ...timestamps,
});

// Contacts Table
export const contacts = sqliteTable("contacts", {
  ...id,
  name: text("name").notNull(),
  email: text("email").notNull(),
  phone: text("phone"),
  userId: integer("user_id").notNull().references(() => users.id),
  ...timestamps,
});

// Conversations Table
export const conversations = sqliteTable("conversations", {
  ...id,
  title: text("title").notNull(),
  userId: integer("user_id").notNull().references(() => users.id),
  chatbotId: integer("chatbot_id").notNull().references(() => chatbots.id),
  contactId: integer("contact_id").notNull().references(() => contacts.id),
  ...timestamps,
});

// Messages Table
export const messages = sqliteTable("messages", {
  ...id,
  text: text("text").notNull(),
  conversationId: integer("conversation_id").notNull().references(() => conversations.id),
  userId: integer("user_id").notNull().references(() => users.id),
  ...timestamps,
});

export const messagesRelations = relations(messages, ({ one }) => ({
  conversation: one(conversations, {
    fields: [messages.conversationId],
    references: [conversations.id],
  }),
  user: one(users, {
    fields: [messages.userId],
    references: [users.id],
  }),
}));

export const usersRelations = relations(users, ({ many }) => ({
  chatbots: many(chatbots),
  contacts: many(contacts),
  conversations: many(conversations),
  messages: many(messages),
}));

export const chatbotsRelations = relations(chatbots, ({ one, many }) => ({
  user: one(users, {
    fields: [chatbots.userId],
    references: [users.id],
  }),
  conversations: many(conversations),
}));

export const contactsRelations = relations(contacts, ({ one, many }) => ({
  user: one(users, {
    fields: [contacts.userId],
    references: [users.id],
  }),
  conversations: many(conversations),
}));

export const conversationsRelations = relations(conversations, ({ one, many }) => ({
  user: one(users, {
    fields: [conversations.userId],
    references: [users.id],
  }),
  chatbot: one(chatbots, {
    fields: [conversations.chatbotId],
    references: [chatbots.id],
  }),
  contact: one(contacts, {
    fields: [conversations.contactId],
    references: [contacts.id],
  }),
  messages: many(messages),
}));

export const selectTasksSchema = createSelectSchema(tasks);

export const insertTasksSchema = createInsertSchema(
  tasks,
  {
    name: schema => schema.name.min(1).max(500),
  },
).required({
  done: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const registerUserSchema = createInsertSchema(
  users,
  {
    name: schema => schema.name.min(1).max(500),
    email: schema => schema.email.email().min(1).max(500),
    password: schema => schema.password.min(8).max(500),
  },
).required({
  email: true,
  password: true,
  name: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const selectChatbotsSchema = createSelectSchema(chatbots);
export const insertChatbotsSchema = createInsertSchema(
  chatbots,
  {
    title: schema => schema.title.min(1).max(100),
    description: schema => schema.description.min(1).max(500),
    commandTemplate: schema => schema.commandTemplate.min(1).max(1000),
    modelAi: schema => schema.modelAi.min(1).max(100),
    embedingModel: schema => schema.embedingModel.min(1).max(100),
    sugestionMessage: schema => schema.sugestionMessage.min(1).max(1000),
    pdfTitle: schema => schema.pdfTitle.min(1).max(500),
    pdfLink: schema => schema.pdfLink.min(1).max(500),
  },
).required({
  title: true,
  description: true,
  commandTemplate: true,
  modelAi: true,
  embedingModel: true,
  sugestionMessage: true,
  pdfTitle: true,
  pdfLink: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  userId: true,
});

export const patchChatbotsSchema = insertChatbotsSchema.partial();

export const selectContactsSchema = createSelectSchema(contacts);

export const loginUserSchema = registerUserSchema.omit({
  name: true,
});

export const patchTasksSchema = insertTasksSchema.partial();
