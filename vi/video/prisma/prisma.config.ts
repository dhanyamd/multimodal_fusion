// Prisma 7 configuration file
// Connection URL is now configured here instead of schema.prisma
import { defineConfig } from "prisma/config";

export default defineConfig({
  schema: "./prisma/schema.prisma",
  datasource: {
    url: process.env.DATABASE_URL,
  },
});

