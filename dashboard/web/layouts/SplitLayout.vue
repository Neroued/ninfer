<template>
  <div class="stage">
    <header class="topbar"><slot name="header" /></header>
    <slot name="notice" />
    <main class="split-main">
      <div class="split-left">
        <slot name="hero" /><slot name="timeline" /><slot name="requests" />
      </div>
      <slot name="rail" />
    </main>
    <footer class="page-footer mono"><slot name="footer" /></footer>
  </div>
  <slot name="overlay" />
</template>
