<script setup lang="ts">
import axios from "axios";
import { computed, onMounted, ref } from "vue";

import { getAppRoot } from "@/onload/loadConfig";

const jobs = ref<any[]>([]);

const jobCount = computed(() => jobs.value.length);

onMounted(async () => {
	const path = `${getAppRoot()}api/jobs`;
	try {
    	const response = await axios.get(path);
    	console.log(`${response.data.length} jobs loaded.`);
    	jobs.value = response.data;
} catch (error) {
    	console.error(error);
	}
});
</script>

<template>
	<div>
    	<h1>Jobs</h1>
    	<p>{{ jobCount }} jobs loaded.</p>

    	<ul>
        	<li v-for="job in jobs" :key="job.id">
            	{{ job }}
        	</li>
    	</ul>
	</div>
</template>
