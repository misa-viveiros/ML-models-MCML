import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk

def main():
    """
    Main function to process data, create plots, and animate the data.
    """
    data_num = 11
    full = ""  # "full_"

    def process_data(input_file_path, output_file_path):
        """
        Processes the input data file and writes the modified data to the output file.

        Args:
            input_file_path (str): Path to the input data file.
            output_file_path (str): Path to the output data file.
        """
        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        if "full" not in input_file_path:
            # Remove the first 14 rows // 11 rows for hetero data!!
            lines = lines[11:]

        # Split the lines into columns
        data = [line.split() for line in lines]

        if "full" not in input_file_path:
            # Replace the last 3 entries of the 1st column with the 4th-to-last value
            if len(data) >= 4:
                replacement_value = data[-4][0]
                for i in range(1, 4):
                    data[-i][0] = replacement_value

        lines = [' '.join(row) + '\n' for row in data]

        with open(output_file_path, 'w') as file:
            file.writelines(lines)

    process_data(f'hetero_filtered_data_{data_num}_{full}actual_values.txt', f'hetero_filtered_modified_data_{data_num}_{full}actual_values.txt')
    process_data(f'hetero_filtered_data_{data_num}_{full}predicted_values.txt', f'hetero_filtered_modified_data_{data_num}_{full}predicted_values.txt')

    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/Hirom/Downloads/ffmpeg-7.1-essentials_build/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe'
    actual_data = np.loadtxt(f'hetero_filtered_modified_data_{data_num}_{full}actual_values.txt')
    predicted_data = np.loadtxt(f'hetero_filtered_modified_data_{data_num}_{full}predicted_values.txt')

    t_values = actual_data[:, 0]
    u_micro_actual = actual_data[:, 1]
    v_micro_actual = actual_data[:, 2]
    a_micro_actual = actual_data[:, 3]

    u_micro_predicted = predicted_data[:, 1]
    v_micro_predicted = predicted_data[:, 2]
    a_micro_predicted = predicted_data[:, 3]

    num_points = 17

    u_micro_actual_chunks = [u_micro_actual[i:i + num_points] for i in range(0, len(u_micro_actual), num_points)]
    v_micro_actual_chunks = [v_micro_actual[i:i + num_points] for i in range(0, len(v_micro_actual), num_points)]
    a_micro_actual_chunks = [a_micro_actual[i:i + num_points] for i in range(0, len(a_micro_actual), num_points)]
    u_micro_predicted_chunks = [u_micro_predicted[i:i + num_points] for i in range(0, len(u_micro_predicted), num_points)]
    v_micro_predicted_chunks = [v_micro_predicted[i:i + num_points] for i in range(0, len(v_micro_predicted), num_points)]
    a_micro_predicted_chunks = [a_micro_predicted[i:i + num_points] for i in range(0, len(a_micro_predicted), num_points)]
    t_values_chunks = [t_values[i] for i in range(0, len(t_values), num_points)]

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    fig, axs = plt.subplots(3, 1, figsize=(screen_width / 100, screen_height / 100))
    for ax in axs.flat:
        ax.set_xlim(0, num_points + 1)
        ax.set_xticks(range(1, num_points + 1))

    # Set y-limits to fit all points initially
    axs[0].set_ylim(np.min([u_micro_actual, u_micro_predicted]), np.max([u_micro_actual, u_micro_predicted]))
    axs[0].set_ylabel('u_micro')
    line_u_actual, = axs[0].plot([], [], 'bo-', label='Actual', alpha=0.7)
    line_u_predicted, = axs[0].plot([], [], 'ro-', label='Predicted', alpha=0.7)

    axs[1].set_ylim(np.min([v_micro_actual, v_micro_predicted]), np.max([v_micro_actual, v_micro_predicted]))
    axs[1].set_ylabel('v_micro')
    line_v_actual, = axs[1].plot([], [], 'bo-', label='Actual', alpha=0.7)
    line_v_predicted, = axs[1].plot([], [], 'ro-', label='Predicted', alpha=0.7)

    axs[2].set_ylim(np.min([a_micro_actual, a_micro_predicted]), np.max([a_micro_actual, a_micro_predicted]))
    axs[2].set_ylabel('a_micro')
    line_a_actual, = axs[2].plot([], [], 'bo-', label='Actual', alpha=0.7)
    line_a_predicted, = axs[2].plot([], [], 'ro-', label='Predicted', alpha=0.7)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()

    def init():
        """
        Initializes the animation by setting empty data for all lines.

        Returns:
            tuple: A tuple containing all line objects.
        """
        line_u_actual.set_data([], [])
        line_v_actual.set_data([], [])
        line_a_actual.set_data([], [])
        line_u_predicted.set_data([], [])
        line_v_predicted.set_data([], [])
        line_a_predicted.set_data([], [])
        return line_u_actual, line_v_actual, line_a_actual, line_u_predicted, line_v_predicted, line_a_predicted

    def animate(i):
        """
        Updates the animation for frame i.

        Args:
            i (int): The frame index.

        Returns:
            tuple: A tuple containing all line objects.
        """
        if i < len(u_micro_actual_chunks):
            x_data = range(1, num_points + 1)
            line_u_actual.set_data(x_data, u_micro_actual_chunks[i])
            line_v_actual.set_data(x_data, v_micro_actual_chunks[i])
            line_a_actual.set_data(x_data, a_micro_actual_chunks[i])
            line_u_predicted.set_data(x_data, u_micro_predicted_chunks[i])
            line_v_predicted.set_data(x_data, v_micro_predicted_chunks[i])
            line_a_predicted.set_data(x_data, a_micro_predicted_chunks[i])
            current_time = t_values_chunks[i]
            axs[0].set_title(f'2D Animation of u_micro at t={current_time:.16f}')
            axs[1].set_title(f'2D Animation of v_micro at t={current_time:.16f}')
            axs[2].set_title(f'2D Animation of a_micro at t={current_time:.16f}')
        return line_u_actual, line_v_actual, line_a_actual, line_u_predicted, line_v_predicted, line_a_predicted

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(u_micro_actual_chunks), interval=10, blit=False)

    output_path = fr'C:\Users\Hirom\OneDrive\Vanderbilt\Research\Data graphs\Hetero Data {data_num}\animation_data_{data_num}_{full}.mp4'
    ani.save(output_path, writer='ffmpeg', fps=5)

    plt.show()

if __name__ == '__main__':
    main()