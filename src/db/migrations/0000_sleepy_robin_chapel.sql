CREATE TABLE `chatbots` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`title` text NOT NULL,
	`description` text,
	`is_public` integer DEFAULT false NOT NULL,
	`welcome_message` text NOT NULL,
	`suggestion_message` text NOT NULL,
	`system_prompt` text DEFAULT 'AI assistant is a professional and polite customer service work at PT. Omni Hottilier representative. 

The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness. 

AI assistant provides clear, concise, and friendly responses without repeating unnecessary information or phrases such as "Berdasarkan informasi yang diberikan sebelumnya.", "dalam konteks yang diberikan.", "dalam konteks yang tersedia.". 

AI is a well-behaved and well-mannered individual. 

AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user. 

AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation. 

AI assistant make answer using Indonesian Language. 

AI assistant avoids sounding repetitive and ensures responses sound natural and tailored to each question. 

If the context does not provide the answer to question, the AI assistant will say, "Mohon Maaf, tapi saya tidak dapat menjawab pertanyaan tersebut saat ini.". 

AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation. 

AI assistant will not apologize for previous responses, but instead will indicated new information was gained. 

AI assistant will not invent anything that is not drawn directly from the context.
' NOT NULL,
	`ai_model` text DEFAULT 'gpt-3.5-turbo' NOT NULL,
	`is_proposed_model` integer DEFAULT true NOT NULL,
	`embedding_model` text DEFAULT 'fastext' NOT NULL,
	`temperature` integer DEFAULT 30 NOT NULL,
	`max_tokens` integer DEFAULT 500 NOT NULL,
	`pdf_title` text NOT NULL,
	`pdf_link` text NOT NULL,
	`user_id` integer NOT NULL,
	`created_at` integer,
	`updated_at` integer,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `contacts` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`name` text NOT NULL,
	`email` text NOT NULL,
	`phone` text,
	`user_id` integer NOT NULL,
	`created_at` integer,
	`updated_at` integer,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `conversations` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` integer NOT NULL,
	`chatbot_id` integer NOT NULL,
	`contact_id` integer NOT NULL,
	`created_at` integer,
	`updated_at` integer,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`chatbot_id`) REFERENCES `chatbots`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`contact_id`) REFERENCES `contacts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `messages` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`text` text NOT NULL,
	`conversation_id` integer NOT NULL,
	`user_id` integer NOT NULL,
	`is_bot` integer DEFAULT false NOT NULL,
	`created_at` integer,
	`updated_at` integer,
	FOREIGN KEY (`conversation_id`) REFERENCES `conversations`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `model_responses` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`message_id` integer NOT NULL,
	`model_type` text NOT NULL,
	`query` text NOT NULL,
	`processing_time` integer NOT NULL,
	`results` text NOT NULL,
	`metadata` text NOT NULL,
	`complexity_analysis` text,
	`search_pipeline` text,
	`model_approach` text,
	`pipeline_steps` text,
	`gpt_generation` text,
	`ragas_evaluation` text,
	`message` text,
	`user_id` integer NOT NULL,
	`chatbot_id` integer NOT NULL,
	`created_at` integer,
	`updated_at` integer,
	FOREIGN KEY (`message_id`) REFERENCES `messages`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`chatbot_id`) REFERENCES `chatbots`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `queryProposedModelResponses` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`message_id` integer NOT NULL,
	`created_at` integer,
	`updated_at` integer,
	FOREIGN KEY (`message_id`) REFERENCES `messages`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `tasks` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`name` text NOT NULL,
	`done` integer DEFAULT false NOT NULL,
	`created_at` integer,
	`updated_at` integer
);
--> statement-breakpoint
CREATE TABLE `users` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`name` text NOT NULL,
	`password` text NOT NULL,
	`email` text NOT NULL,
	`created_at` integer,
	`updated_at` integer
);
--> statement-breakpoint
CREATE UNIQUE INDEX `model_responses_message_id_unique` ON `model_responses` (`message_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `emailUniqueIndex` ON `users` (lower("email"));