<template>
  <div v-if="type === 'card'" class="sk-card">
    <div class="sk-line sk-title" />
    <div class="sk-line sk-body" />
    <div class="sk-line sk-body-short" />
  </div>
  <div v-else-if="type === 'stats'" class="sk-stats-row">
    <div v-for="i in rows" :key="i" class="sk-stat-block">
      <div class="sk-line sk-stat-title" />
      <div class="sk-line sk-stat-value" />
    </div>
  </div>
  <div v-else class="sk-table">
    <div class="sk-row sk-header-row">
      <div v-for="i in 4" :key="'h' + i" class="sk-line sk-col" />
    </div>
    <div v-for="row in rows" :key="row" class="sk-row">
      <div
        v-for="col in 4"
        :key="'r' + row + 'c' + col"
        class="sk-line sk-col"
        :class="{ 'sk-col-short': col === 4 }"
      />
    </div>
  </div>
</template>

<script setup>
defineProps({
  type: {
    type: String,
    default: 'table',
    validator: (v) => ['table', 'card', 'stats'].includes(v),
  },
  rows: { type: Number, default: 5 },
})
</script>

<style scoped>
.sk-line {
  background: linear-gradient(
    90deg,
    var(--bg-elevated) 25%,
    var(--bg-overlay) 50%,
    var(--bg-elevated) 75%
  );
  background-size: 200px 100%;
  animation: shimmer 1.5s infinite;
  border-radius: var(--radius-sm);
}

/* Card */
.sk-card {
  padding: 20px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
}

.sk-title {
  height: 16px;
  width: 60%;
  margin-bottom: 16px;
}

.sk-body {
  height: 12px;
  width: 100%;
  margin-bottom: 8px;
}

.sk-body-short {
  height: 12px;
  width: 75%;
}

/* Stats row */
.sk-stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.sk-stat-block {
  padding: 16px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
}

.sk-stat-title {
  height: 12px;
  width: 40%;
  margin-bottom: 10px;
}

.sk-stat-value {
  height: 28px;
  width: 55%;
}

/* Table */
.sk-table {
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
  padding: 16px;
}

.sk-row {
  display: flex;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px solid var(--border-subtle);
}

.sk-row:last-child {
  border-bottom: none;
}

.sk-header-row {
  padding-bottom: 16px;
  border-bottom: 2px solid var(--border-subtle);
}

.sk-col {
  flex: 1;
  height: 14px;
}

.sk-col-short {
  flex: 0.5;
}

@keyframes shimmer {
  0% {
    background-position: -200px 0;
  }
  100% {
    background-position: calc(200px + 100%) 0;
  }
}
</style>
