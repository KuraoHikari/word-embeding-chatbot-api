import type { SQL } from "drizzle-orm";
import type { AnySQLiteColumn } from "drizzle-orm/sqlite-core";

import { relations, sql } from "drizzle-orm";
import { integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";

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
  isPublic: integer("is_public", { mode: "boolean" }).notNull().default(false),
  welcomeMessage: text("welcome_message").notNull(),
  suggestionMessage: text("suggestion_message").notNull(),
  systemPrompt: text("system_prompt").notNull(),
  aiModel: text("ai_model").notNull(),
  isProposedModel: integer("is_proposed_model", { mode: "boolean" }).notNull().default(true),
  embeddingModel: text("embedding_model").notNull().default("fastext"),
  temperature: integer("temperature").notNull().default(0.3),
  maxTokens: integer("max_tokens").notNull().default(500),
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
  isBot: integer("is_bot", { mode: "boolean" }).notNull().default(false),
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
    isPublic: schema => schema.isPublic,
    welcomeMessage: schema => schema.welcomeMessage.min(1).max(1000),
    suggestionMessage: schema => schema.suggestionMessage.min(1).max(1000),
    systemPrompt: schema => schema.systemPrompt.min(1).max(2000),
    aiModel: schema => schema.aiModel.min(1).max(500),
    isProposedModel: schema => schema.isProposedModel,
    embeddingModel: schema => schema.embeddingModel.min(1).max(500),
    temperature: schema => schema.temperature.min(0).max(1),
    maxTokens: schema => schema.maxTokens.min(1).max(4000),
    pdfTitle: schema => schema.pdfTitle.min(1).max(500),
    pdfLink: schema => schema.pdfLink.min(1).max(500),
  },
).required({
  title: true,
  description: true,
  isPublic: true,
  welcomeMessage: true,
  suggestionMessage: true,
  systemPrompt: true,
  aiModel: true,
  isProposedModel: true,
  embeddingModel: true,
  temperature: true,
  maxTokens: true,
  pdfTitle: true,
  pdfLink: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  userId: true,
});

export const patchChatbotsSchema = insertChatbotsSchema.omit({
  pdfTitle: true,
  pdfLink: true,
}).partial();

export const selectContactsSchema = createSelectSchema(contacts);

export const loginUserSchema = registerUserSchema.omit({
  name: true,
});

export const patchTasksSchema = insertTasksSchema.partial();

export const selectContactSchema = createSelectSchema(contacts);

export const createContactSchema = createInsertSchema(
  contacts,
  {
    name: schema => schema.name.min(1).max(500),
    email: schema => schema.email.email().min(1).max(500),
    phone: schema => schema.phone.min(1).max(500),
  },
).required({
  name: true,
  email: true,
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  userId: true,
}).partial({
  phone: true,
});

export const patchContactSchema = createContactSchema.partial();
export const selectConversationsSchema = createSelectSchema(conversations);
export const insertConversationsSchema = createInsertSchema(
  conversations,
  {
    chatbotId: schema => schema.chatbotId.min(1).max(500),
  },
).required({
  chatbotId: true,
}).omit({
  id: true,
  contactId: true,
  userId: true,
  createdAt: true,
  updatedAt: true,
});

export const selectMessagesSchema = createSelectSchema(messages);
export const insertMessagesSchema = createInsertSchema(
  messages,
  {
    text: schema => schema.text.min(1).max(500),
    conversationId: schema => schema.conversationId.min(1).max(500),
  },
).required({
  text: true,
}).omit({
  id: true,
  userId: true,
  createdAt: true,
  updatedAt: true,
  isBot: true,
}).extend({
  // extend chatbotId required
  chatbotId: z.number().min(1).max(500),
});
