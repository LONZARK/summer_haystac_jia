import os
import io
import json
import sys
import argparse
import pstats
import cProfile
import time

from haystac_kw.ta1.agent_behavior_sim.agent_behavior_sim import AgentBehaviorSim

def get_options(cmd_args):
    """Argument Parser.

    Parameters
    ----------
    cmd_args : argparse.ArgumentParser
        Input arguments
    """
    parser = argparse.ArgumentParser(
        prog="agent_itinerary_sim.py",
        usage="%(prog)s -c configuration.json",
        description="SUMO Agent Mobility Simulation",
    )
    parser.add_argument(
        "-c",
        type=str,
        dest="config",
        required=True,
        help="JSON configuration file.")
    parser.add_argument(
        "--profiling",
        dest="profiling",
        action="store_true",
        help="Enable Python3 cProfile feature.",
    )
    parser.add_argument(
        "--no-profiling",
        dest="profiling",
        action="store_false",
        help="Disable Python3 cProfile feature.",
    )

    parser.set_defaults(profiling=False)
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """Simulate agent itinerary.

    Parameters
    ----------
    cmd_args : argparse.ArgumentParser
        Input arguments
    """
    tic = time.time()
    args = get_options(cmd_args)

    # Profiler.
    if args.profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    conf_fname = args.config
    os.chdir(os.path.split(conf_fname)[0])

    print("Loading configuration file {}.".format(conf_fname))
    conf = json.loads(open(conf_fname).read())
    profiling = args.profiling
    sim = AgentBehaviorSim(conf, profiling=profiling)
    sim.create_agents()

    sim.simulate()

    # Profiler.
    if args.profiling:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats(
            "cumulative").print_stats(25)
        print(results.getvalue())

    if not sim.sumo_gui:
        sim.close()

    print('Total simulation took %0.3f minutes' % ((time.time() - tic)/60))


if __name__ == "__main__":
    main(sys.argv[1:])
