<template>
  <div class="app-frame">
    <div class="stage">
      <header class="topbar"><slot name="header" /></header>
      <slot name="notice" />
      <main class="split-main">
        <div class="split-left">
          <slot name="hero" />
          <div class="timeline-block"><slot name="timeline" /></div>
          <slot name="requests" />
        </div>
        <div class="rail-shell"><slot name="rail" /></div>
      </main>
      <footer class="page-footer mono"><slot name="footer" /></footer>
    </div>
    <slot name="overlay" />
  </div>
</template>
