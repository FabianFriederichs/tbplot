import numpy as np
import pandas as pd
import argparse
import os
import json
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from cycler import cycler

def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def print_event_file_info(event_file: str, accumulator: EventAccumulator):
    # print tree of tags
    tags = accumulator.Tags()
    print(f"Event File: {event_file}")
    for tag, subtags in tags.items():        
        if is_iterable(subtags):
            print(f"   +- {tag}")
            for subtag in subtags:
                print(f"   |  +- {subtag}")
        else:
            print(f"   +- {tag}:\t{str(subtags)}")

def scalars_to_dataframe(accumulator: EventAccumulator):
    # return a dictionary of dataframes, one for each tag
    dataframes = {}
    for tag in accumulator.Tags()['scalars']:
        dataframes[tag] = pd.DataFrame(accumulator.Scalars(tag))
    return dataframes

# plotting strategies
def plot_all_runs(tag_index, fig, axes, steps, values, plot_data, settings):
    # set axis labels
    axes.set_xlabel(plot_data['tag_x_labels'][tag_index], fontsize = settings['fontsize']['axis_label'])
    axes.set_ylabel(plot_data['tag_y_labels'][tag_index], fontsize = settings['fontsize']['axis_label'])

    # plot data
    for j in range(len(plot_data['event_files'])):
        axes.plot(
            steps[j], values[j], 
            label = plot_data['run_labels'][j] if plot_data['run_labels'] is not None else plot_data['event_files'][j],
            linewidth = settings['linestyles']['primary']['width'],
            linestyle = settings['linestyles']['primary']['style'],
            alpha = settings['linestyles']['primary']['alpha']
        )

    # add legend
    if settings['legend']:
        axes.legend(fontsize = settings['fontsize']['legend'])

def plot_all_runs_plus_average(tag_index, fig, axes, steps, values, plot_data, settings):
    # set axis labels
    axes.set_xlabel(plot_data['tag_x_labels'][tag_index], fontsize = settings['fontsize']['axis_label'])
    axes.set_ylabel(plot_data['tag_y_labels'][tag_index], fontsize = settings['fontsize']['axis_label'])

    # calculate average
    average_steps = steps[0]
    average_values = np.zeros(len(average_steps))
    for i in range(len(plot_data['event_files'])):
        average_values += np.interp(average_steps, steps[i], values[i])
    average_values /= len(plot_data['event_files'])

    # plot data
    for j in range(len(plot_data['event_files'])):
        axes.plot(
            steps[j], values[j], 
            label = plot_data['run_labels'][j] if plot_data['run_labels'] is not None else plot_data['event_files'][j],
            linewidth = settings['linestyles']['secondary']['width'],
            linestyle = settings['linestyles']['secondary']['style'],
            alpha = settings['linestyles']['secondary']['alpha']
        )

    # plot average
    axes.plot(
        average_steps, average_values, 
        label = plot_data['average_label'],
        linewidth = settings['linestyles']['primary']['width'],
        linestyle = settings['linestyles']['primary']['style'],
        alpha = settings['linestyles']['primary']['alpha']
    )

    # add legend
    if settings['legend']:
        axes.legend(fontsize = settings['fontsize']['legend'])

def plot_average_only(tag_index, fig, axes, steps, values, plot_data, settings):
    # set axis labels
    axes.set_xlabel(plot_data['tag_x_labels'][tag_index], fontsize = settings['fontsize']['axis_label'])
    axes.set_ylabel(plot_data['tag_y_labels'][tag_index], fontsize = settings['fontsize']['axis_label'])

    # calculate average
    average_steps = steps[0]
    average_values = np.zeros(len(average_steps))
    for i in range(len(plot_data['event_files'])):
        average_values += np.interp(average_steps, steps[i], values[i])
    average_values /= len(plot_data['event_files'])
    
    # plot data
    axes.plot(
        average_steps, average_values, 
        label = plot_data['average_label'],
        linewidth = settings['linestyles']['primary']['width'],
        linestyle = settings['linestyles']['primary']['style'],
        alpha = settings['linestyles']['primary']['alpha']
    )

    # add legend
    if settings['legend']:
        axes.legend(fontsize = settings['fontsize']['legend'])

# main plot function
def plot_tensorboard(plot_data):
    # get plot settings
    settings = plot_data['plot_settings']
    # load event data using tensorboard API
    accumulators = {}
    for f in plot_data['event_files']:
        accumulators[f] = EventAccumulator(f)
        accumulators[f].Reload()

    # print info about event files
    for f, accumulator in accumulators.items():
        print_event_file_info(f, accumulator)

    # get dataframes for each event file
    dataframes = {}
    for f, accumulator in accumulators.items():
        dataframes[f] = scalars_to_dataframe(accumulator)

    # calculate plot specs
    if plot_data['plot_specs'] is None:
        nrow = len(plot_data['tags'])
        ncol = 1
        # fill grid specs
        grid_specs = []
        for i, tag in enumerate(plot_data['tags']):
            grid_specs.append(((i, 0), (1, 1)))
    else:
        # parse plot specs ( format: <startrow>:<startcol>:<numrows>:<numcols> )
        grid_specs = []
        for spec in plot_data['plot_specs']:
            parsed = spec.split(':')
            s = ((int(parsed[0]), int(parsed[1])),(int(parsed[2]), int(parsed[3])))
            # check if valid grid spec
            if s[0][0] < 0 or s[0][1] < 0 or s[1][0] < 1 or s[1][1] < 1:
                raise ValueError(f"Invalid grid spec: {spec}")
            grid_specs.append(((int(parsed[0]), int(parsed[1])),(int(parsed[2]), int(parsed[3]))))

        # lambda, checks if two grid specs overlap
        def overlap(a, b):
            return not (
                a[0][0] + a[1][0] <= b[0][0] or b[0][0] + b[1][0] <= a[0][0] or
                a[0][1] + a[1][1] <= b[0][1] or b[0][1] + b[1][1] <= a[0][1]
            )        
        
        # check if grid specs overlap
        for i in range(len(grid_specs)):
            for j in range(i + 1, len(grid_specs)):
                if overlap(grid_specs[i], grid_specs[j]):
                    raise ValueError(f"Grid specs overlap: {plot_data['plot_specs'][i]} and {plot_data['plot_specs'][j]}")

        # calculate max grid size
        nrow = 0
        ncol = 0
        for spec in grid_specs:
            nrow = max(nrow, int(spec[0][0]) + int(spec[1][0]))
            ncol = max(ncol, int(spec[0][1]) + int(spec[1][1]))

    # create figure with grid spec        
    fig = plt.figure()
    fig.set_size_inches(settings["width"] / 2.54, settings["height"] / 2.54)
    fig.set_constrained_layout(True)
    gs = fig.add_gridspec(nrow, ncol)

    # create axes for each tag
    axes = {}
    for i, tag in enumerate(plot_data['tags']):
        axes[tag] = fig.add_subplot(
            gs[grid_specs[i][0][0]:grid_specs[i][0][0] + grid_specs[i][1][0],
                grid_specs[i][0][1]:grid_specs[i][0][1] + grid_specs[i][1][1]]
        )
        # set axis title
        axes[tag].set_title(tag if plot_data['tag_labels'] is None else plot_data['tag_labels'][i], fontsize = settings['fontsize']['plot_title'])
        # set color cycle if specified
        if 'color_cycle' in settings:
            axes[tag].set_prop_cycle(cycler(color=settings['color_cycle']))
        # set tick label size
        axes[tag].tick_params(axis='both', which='major', labelsize=settings['fontsize']['axis_tick_major'])
        axes[tag].tick_params(axis='both', which='minor', labelsize=settings['fontsize']['axis_tick_minor'])
        # tick label format
        axes[tag].ticklabel_format(axis='x', style=settings['tick_style_x'], scilimits=settings['tick_sci_limits']['x'])
        axes[tag].ticklabel_format(axis='y', style=settings['tick_style_y'], scilimits=settings['tick_sci_limits']['y'])

    # plot data
    for i, tag in enumerate(plot_data['tags']):
        # convert data to numpy arrays
        steps = [dataframes[f][tag].loc[:,'step'].to_numpy() for f in plot_data['event_files']]
        values = [dataframes[f][tag].loc[:,'value'].to_numpy() for f in plot_data['event_files']]

        if plot_data['plot_mode'] == 'all_runs_plus_average':
            plot_all_runs_plus_average(i, fig, axes[tag], steps, values, plot_data, settings)
        elif plot_data['plot_mode'] == 'average_only':
            plot_average_only(i, fig, axes[tag], steps, values, plot_data, settings)
        elif plot_data['plot_mode'] == 'all_runs':
            plot_all_runs(i, fig, axes[tag], steps, values, plot_data, settings)
    
    # save figure
    fig.savefig(plot_data['output'], dpi=300)

    # if show is true, show plot
    if plot_data['show']:
        plt.show(block = True)

def main():
    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_files', nargs='+', help='a list of event file paths', required=True)
    # info mode
    parser.add_argument('--info', help='Prints info about the event files.', action='store_true')
    # plot mode
    parser.add_argument('--plot', help='Makes a plot of the supplied data.', action='store_true')
    parser.add_argument('--plot_mode', help='Selects the plot mode.', default='all_runs_plus_average', choices=['all_runs', 'all_runs_plus_average', 'average_only'])
    parser.add_argument('--average_label', help='Label for the average plot for the plot modes \'all_runs_plus_average\' and \'average_only\'', default='Average')
    parser.add_argument('--show', help='If left out, nothing is shown and the figure is immediately saved to disk.', action='store_true')
    parser.add_argument('--run_labels', nargs='+', help='How the different runs corresponding to the event files should be labeled. If left out, the event file names are used.')
    parser.add_argument('--tags', nargs='+', help='Which tags to plot, in that order.')
    parser.add_argument('--tag_labels', nargs='+', help='A list of labels for the tags to plot. If left out, the tag names are used.') 
    parser.add_argument('--tag_x_labels', nargs='+', help='X-Axis labels for the tag plots.', default='Step') 
    parser.add_argument('--tag_y_labels', nargs='+', help='Y-Axis labels for the tag plots.', default='Value')  
    parser.add_argument('--plot_specs', nargs='+', help='Specifies the plot position and size of each tag plot. The format is <startrow>:<startcolumn>:<numrows>:<numcolumns>. The ranges should not overlap.', type=str, required=False)
    parser.add_argument('--output', help='Output file path. The extension determines the format that will be saved.', default='plot.pdf')
    parser.add_argument('--plot_title', help='Main title of the figure.', default='Untitled Plot')
    parser.add_argument('--settings', help='Path to the plot settings file.', default="plot_settings.json")

    # parse arguments
    args = parser.parse_args()

    # load plot settings
    if args.settings is not None:
        if not os.path.exists(args.settings):
            raise ValueError(f"Settings file not found: {args.settings}")
        with open(args.settings) as settings_file:
            settings = json.load(settings_file)

    # check if event file exists and throw error if not found
    for f in args.event_files:
        if not os.path.exists(f):
            raise ValueError(f"Event file not found: {f}")

    # load event data using tensorboard API
    accumulators = {}
    for f in args.event_files:
        accumulators[f] = EventAccumulator(f)
        accumulators[f].Reload()

    if args.info:
        for f, accumulator in accumulators.items():
            print_event_file_info(f, accumulator)
    else:
        if args.tags is None:
            raise ValueError("No tags specified")
        if args.plot_specs is not None and len(args.plot_specs) != len(args.tags):
            raise ValueError("Number of plot specs does not match number of tags")
        if args.tag_labels is not None and len(args.tag_labels) != len(args.tags):
            raise ValueError("Number of tag labels does not match number of tags")
        if args.run_labels is not None and len(args.run_labels) != len(args.event_files):
            raise ValueError("Number of run labels does not match number of event files")
        
        result = plot_tensorboard({
            "event_files": args.event_files,
            "run_labels": args.run_labels,
            "tags": args.tags,
            "tag_labels": args.tag_labels,
            "tag_x_labels": args.tag_x_labels,
            "tag_y_labels": args.tag_y_labels,
            "plot_specs": args.plot_specs,
            "output": args.output,
            "plot_mode": args.plot_mode,
            "average_label": args.average_label,
            "plot_title": args.plot_title,
            "show": args.show,
            "plot_settings": settings
        })
        if args.show:
            plt.show(block = True)

if __name__ == '__main__':
    main()