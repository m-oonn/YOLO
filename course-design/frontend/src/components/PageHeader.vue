<template>
  <div class="page-header">
    <div class="header-info">
      <div class="header-accent" />
      <div class="header-text">
        <h2 class="header-title">{{ title }}</h2>
        <p v-if="subtitle" class="header-subtitle">{{ subtitle }}</p>
      </div>
    </div>
    <div v-if="$slots.actions" class="header-actions">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, default: '' },
})
</script>

<style scoped>
.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 24px;
  gap: 16px;
}

.header-info {
  display: flex;
  align-items: flex-start;
  gap: 14px;
}

.header-accent {
  width: 4px;
  height: 40px;
  border-radius: 2px;
  background: linear-gradient(180deg, var(--color-primary), var(--color-primary) 60%, transparent);
  flex-shrink: 0;
  margin-top: 2px;
}

.header-text {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.header-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.3;
  margin: 0;
  letter-spacing: 0.3px;
}

.header-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 0;
  line-height: 1.4;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    gap: 12px;
  }
  .header-actions {
    width: 100%;
  }
  .header-actions :deep(.el-button) {
    flex: 1;
  }
}
</style>
