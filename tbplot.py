import json
import argparse
import plot_tensorboard as ptb

def main():
    # load job file
    parser = argparse.ArgumentParser()
    parser.add_argument("job_file", help="Path to the job file")
    args = parser.parse_args()

    with open(args.job_file, "r") as f:
        jobfile = json.load(f)

    # process jobs
    numjobs = len(jobfile["plot_jobs"])
    for i, job in enumerate(jobfile["plot_jobs"]):
        print(f"Processing job {str(i)}/{str(numjobs)}: {job['job_name']}...")
        ptb.plot_tensorboard({
            "event_files": job["event_files"],
            "run_labels": job["run_labels"],
            "tags": job["tags"],
            "tag_labels": job["tag_labels"],
            "tag_x_labels": job["tag_x_labels"],
            "tag_y_labels": job["tag_y_labels"],
            "plot_specs": job["plot_specs"],
            "output": job["output"],
            "plot_mode": job["plot_mode"],
            "average_label": job["average_label"],
            "plot_title": job["plot_title"],
            "show": job["show"],
            "plot_settings": jobfile["plot_settings"]
        })

if __name__ == "__main__":
    main()