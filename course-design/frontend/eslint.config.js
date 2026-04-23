import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import prettierConfig from '@vue/eslint-config-prettier'
import globals from 'globals'

export default [
  { name: 'app/files-to-ignore', ignores: ['**/dist/**', '**/coverage/**', '**/node_modules/**'] },
  { name: 'app/files-to-lint', files: ['**/*.{js,mjs,vue}'] },
  js.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.es2022 },
    },
  },
  {
    rules: {
      'vue/multi-word-component-names': ['error', { ignores: ['index', 'default'] }],
      'vue/no-v-html': 'warn',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
  prettierConfig,
]
