/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly REACT_APP_GEMINI_API_KEY: string
    // add more env variables here...
  }
  
  interface ImportMeta {
    readonly env: ImportMetaEnv
  }