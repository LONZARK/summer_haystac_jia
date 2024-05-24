import os
import pandas as pd
from multiprocessing import Queue

class ULLTLog:
    """Class for logging ULLT data live"""
    def __init__(self, agent_id, output_folder, checkpoint_rate=86400,
                 writer_queue=None):
        """
        Constructor method

        Parameters
        ----------
        agent_id : str
            string identifying the agent
        output_folder : str
            folder to save logs to
        checkpoint_rate : int, optional
            determines max length of record before writing to disk
            (defaults to 86400)
        writer_queue : multiprocessing.Queue, optional
            queue for writing data to disk in a separate thread
            (defaults to None)
        """
        self.data = {
            'timestamp': [],
            'longitude': [],
            'latitude': []
        }
        # how often to save out checkpoints to disk
        self.checkpoint_rate = checkpoint_rate

        # vars for creating checkpoint files
        self.output_folder = output_folder
        self.checkpoint_num = 0
        self.agent_id = agent_id
        self.writer_queue = writer_queue
        self.checkpointed_logs = []
        os.makedirs(output_folder, exist_ok=True)

    def add_sample(self, dtime, pos):
        """Add a data point to the log

        Parameters
        ----------
        dtime : datetime.datetime
            timetamp of the event
        pos : tuple(float)
            long/lat position
        """
        # Add timestep ULLT data
        self.data['timestamp'].append(dtime)
        self.data['longitude'].append(pos[0])
        self.data['latitude'].append(pos[1])

        if len(self.data['timestamp']) == self.checkpoint_rate:
            # We have enough data points, save to disk
            self.save()

    def save(self):
        """Save out data to parquet"""
        save_file = os.path.join(
            self.output_folder,
            f'{self.agent_id}.{self.checkpoint_num}.parquet')

        os.makedirs(os.path.split(save_file)[0], exist_ok=True)
        if self.writer_queue is not None:
            self.writer_queue.put((save_file, self.data.copy()))
            if len(self.data['timestamp']) > 0:
                self.data = {k: [] for k in self.data.keys()}
                self.checkpoint_num += 1
                self.checkpointed_logs.append(save_file)
            return
        df = pd.DataFrame(self.data)
        if len(df) > 0:
            df.to_parquet(save_file)
            self.checkpoint_num += 1
            self.checkpointed_logs.append(save_file)
            self.data = {k: [] for k in self.data.keys()}
        else:
            pass
            #print(f'{save_file} would have been empty')

    def get_checkpoints(self):
        """Returns a list of checkpoint files the log has written to

        Returns
        -------
        list(string), string
            - checkpointed_logs: list of files writtent to
            - save_file: file to save consolidated files to
        """
        save_file = os.path.join(
            self.output_folder,
            f'{self.agent_id}.parquet')
        return self.checkpointed_logs, save_file


class StopPointLog(ULLTLog):
    """Logger for logging stop point events live"""
    def __init__(
            self,
            agent_id,
            output_folder,
            checkpoint_rate=1000,
            writer_queue=None):
        """
        Constructor method

        Parameters
        ----------
        agent_id : str
            string identifying the agent
        output_folder : str
            folder to save logs to
        checkpoint_rate : int, optional
            determines max length of record before writing to disk
            (defaults to 1000)
        writer_queue : multiprocessing.Queue, optional
            queue for writing data to disk in a separate thread
            (defaults to None)
        """
        super(
            StopPointLog,
            self).__init__(
            agent_id,
            output_folder,
            checkpoint_rate=checkpoint_rate,
            writer_queue=writer_queue)
        self.data = {
            'timestamp': [],
            'longitude': [],
            'latitude': [],
            'event_type': [],
            'event_label': []
        }

    def add_sample(self, timestep, pos, event_type, event_label=None):
        """Add a data point to the log

        Parameters
        ----------
        dtime : datetime.datetime
            timetamp of the event
        pos : tuple(float)
            long/lat position
        event_type : string
            nature of the event (arrival or departure)
        event_label : string, optional
            label for location where the event occured (defaults to None)
        """
        # Add timestep ULLT data
        self.data['event_type'].append(event_type)
        self.data['event_label'].append(event_label)
        super().add_sample(timestep, pos)


def log_writer(queue: Queue):
    """Utility function for using ULLTLogger with multiprocessing.
    Initiate a Pool with this function and write the logs in
    separate processes

    Parameters
    ----------
    queue : Queue
        Queue for tasking log_writer with outputs
    """
    while True:
        args = queue.get(block=True)
        if args is None:
            break
        save_file, data = args
        os.makedirs(os.path.split(save_file)[0], exist_ok=True)
        df = pd.DataFrame(data)
        if len(df) > 0:
            df.to_parquet(save_file)
        else:
            pass


def log_consolidator(args):
    """Utility function for consolidating parquet files
    into a single parquet file.

    Parameters
    ----------
    args : tuple, Using tuple to make use with a Pool easier.
        args = (log_files, output_file)
            - log_files : list of parquet files to combine
            - output_file : file to write the combined files to
    """
    log_files, output_file = args
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)
    dfs = [pd.read_parquet(x) for x in log_files if os.path.isfile(x)]
    if len(dfs) == 0:
        return
    dfs = pd.concat(dfs)
    dfs.to_parquet(output_file)
    if os.path.isfile(output_file):
        # Output file exists, we can
        # delete the input files
        for log_file in log_files:
            if log_file != output_file:
                # Make sure we don't erase
                # the files if it is called
                # on the directory twice
                os.remove(log_file)
