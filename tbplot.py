import json
import argparse
import plot_tensorboard as ptb
import networkx as nx
import datetime as dt

# constants
c_local_job_placeholder_keys = [
    "name",
    "output"
]

# global placeholders
def get_global_placeholders(timestamp):
    return {
        "year" : str(timestamp.year),
        "month" : str(timestamp.month),
        "day" : str(timestamp.day),
        "hour" : str(timestamp.hour),
        "minute" : str(timestamp.minute),
        "second" : str(timestamp.second)
    }

# Apply template to a job. The job can override the template, missing keys are filled in with the template.
def apply_job_template(job, template):
    # apply template
    for key in template:
        if key not in job:
            job[key] = template[key]
        else:
            if isinstance(job[key], dict):
                apply_job_template(job[key], template[key])

# fill string placeholders in jobs
def fill_placeholders(job, global_placeholders):
    # update local placeholders
    local_placeholders = {}
    for key in c_local_job_placeholder_keys:
        local_placeholders[key] = job[key]

    def substitute_local_placeholders(string):
        # replace local placeholders
        for key in local_placeholders:
            string = string.replace(f"${{{key}}}", local_placeholders[key])
        return string
    def substitute_global_placeholders(string):
        # replace global placeholders
        for gkey in global_placeholders:
            string = string.replace(f"${{{gkey}}}", global_placeholders[gkey])
        return string

    # fill placeholders (TODO: General support of nested dicts)
    for key in job:
        if isinstance(job[key], str):
            job[key] = substitute_local_placeholders(job[key])
            job[key] = substitute_global_placeholders(job[key])
        elif isinstance(job[key], list):
            for i, item in enumerate(job[key]):
                if isinstance(item, str):
                    item = substitute_local_placeholders(item)
                    item = substitute_global_placeholders(item)

def load_jobfile(jobfile_path):
    # load job file
    with open(jobfile_path, "r") as f:
        jobfile = json.load(f)

    # process template hierarchy
    # build a graph from the template hierarchy
    graph = nx.DiGraph()
    job_map = {}
    # add job vertices
    for job in jobfile["plot_jobs"]:
        graph.add_node(job['name'])
        job_map[job['name']] = job        

    # add edges
    for i, job in enumerate(jobfile["plot_jobs"]):
        if "template" in job:
            if job["template"] in job_map:
                graph.add_edge(job['template'], job['name'])
            else:
                raise Exception(f"Template \'{job['template']}\' in job \'{job['name']}\' not found!")

    # check for cycles
    if not nx.is_directed_acyclic_graph(graph):
        raise Exception("Job template hierarchy contains cycles!")
    
    # topological sort, then dfs to apply templates to children
    tps = list(nx.topological_sort(graph))

    # apply templates
    for v in tps:
        # if v has successors, apply template
        for u in graph.successors(v):
            # apply template
            apply_job_template(job_map[u], job_map[v])

    # fill placeholders
    global_placeholders = get_global_placeholders(dt.datetime.now())
    for job in jobfile["plot_jobs"]:
        fill_placeholders(job, global_placeholders)

    return jobfile

def main():
    try:
        # load job file
        parser = argparse.ArgumentParser()
        parser.add_argument("job_file", help="Path to the job file")
        parser.add_argument("--print_event_info", help="Print event info", action="store_true")
        args = parser.parse_args()

        # load job file
        jobfile = load_jobfile(args.job_file)

        # process jobs
        numjobs = len(jobfile["plot_jobs"])
        for i, job in enumerate(jobfile["plot_jobs"]):
            print(f"Processing job {str(i + 1)}/{str(numjobs)}: {job['name']}...")
            ptb.plot_tensorboard(
                {
                    "event_files": job["event_files"],
                    "run_labels": job["run_labels"],
                    "tags": job["tags"],
                    "tag_labels": job["tag_labels"],
                    "tag_x_labels": job["tag_x_labels"],
                    "tag_y_labels": job["tag_y_labels"],
                    "grid_specs": job["grid_specs"],
                    "output": job["output"],
                    "plot_mode": job["plot_mode"],
                    "average_label": job["average_label"],
                    "plot_title": job["plot_title"],
                    "show": job["show"],
                    "plot_settings": jobfile["plot_settings"],
                    "plot_settings_overrides": job["plot_settings_overrides"]
                },
                print_info=args.print_event_info
            )
    except Exception as e:
        print(e)
        exit(-1)

if __name__ == "__main__":
    main()