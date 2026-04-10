def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    # Write code here
    task_map = {t["name"]: {"duration": t["duration"], "resources": t["resources"], "dependencies": t["depends_on"]} for t in tasks}

    completed = set()
    running = {}  # task_name -> end_time
    scheduled = []
    current_time = 0
    started = set()

    while len(completed) < len(task_map):
        # Step 1: Complete tasks whose end time has been reached
        for task_name, end_time in list(running.items()):
            if end_time <= current_time:
                completed.add(task_name)
                del running[task_name]

        # Step 2: Identify ready tasks
        ready = [
            name for name, t in task_map.items()
            if name not in started
            and all(dep in completed for dep in t["dependencies"])
        ]

        # Step 3: Sort alphabetically
        ready.sort()

        # Step 4: Greedily assign tasks
        current_resources = sum(task_map[n]["resources"] for n in running)
        for task_name in ready:
            cost = task_map[task_name]["resources"]
            if current_resources + cost <= resource_budget:
                running[task_name] = current_time + task_map[task_name]["duration"]
                scheduled.append((task_name, current_time))
                started.add(task_name)
                current_resources += cost

        if len(completed) == len(task_map):
            break

        # Step 5: Advance time to next completion event
        if running:
            current_time = min(running.values())

    return sorted(scheduled, key=lambda x: (x[1], x[0]))