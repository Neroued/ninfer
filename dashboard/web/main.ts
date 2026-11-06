import { createApp } from 'vue';
import App from './App.vue';
import { initializeTheme } from './theme.js';
import './styles/base.css';
import './styles/split.css';

initializeTheme();
createApp(App).mount('#app');
