import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    """
    Main function to compute and tabulate the normalized L2 norm of LSTM predictions vs actual data.
    """

    def compute_normalized_l2_norm(actual, predicted):
        """
        Computes the normalized L2 norm between actual and predicted data.

        Args:
            actual (numpy.ndarray): The actual data values.
            predicted (numpy.ndarray): The predicted data values.

        Returns:
            float: The normalized L2 norm.
        """
        l2_norm_difference = np.sqrt(np.sum((actual - predicted) ** 2))
        l2_norm_original = np.sqrt(np.sum(actual ** 2))
        return l2_norm_difference / l2_norm_original

    def tabulate_l2_norm():
        """
        Tabulates the normalized L2 norm for displacement, velocity, and acceleration
        across multiple microelements and saves the table to a PDF file.
        """
        elements = [f'Microelement {i}' for i in range(1, 12)]
        rows = ['Displacement', 'Velocity', 'Acceleration']
        l2_norm_table = pd.DataFrame(index=rows, columns=elements)

        for i in range(1, 12):
            actual_file = f'homo_unfiltered_new_data_{i}_actual_values.txt'
            predicted_file = f'homo_unfiltered_new_data_{i}_predicted_values_single_model.txt'

            actual_data = np.loadtxt(actual_file)
            predicted_data = np.loadtxt(predicted_file)

            if actual_data.shape[0] != predicted_data.shape[0]:
                raise ValueError(f'Number of rows do not match for {actual_file} and {predicted_file}')

            u_actual, v_actual, a_actual = actual_data[:, 1], actual_data[:, 2], actual_data[:, 3]
            u_pred, v_pred, a_pred = predicted_data[:, 1], predicted_data[:, 2], predicted_data[:, 3]

            l2_norm_table.at['Displacement', f'Microelement {i}'] = compute_normalized_l2_norm(u_actual, u_pred)
            l2_norm_table.at['Velocity', f'Microelement {i}'] = compute_normalized_l2_norm(v_actual, v_pred)
            l2_norm_table.at['Acceleration', f'Microelement {i}'] = compute_normalized_l2_norm(a_actual, a_pred)

        # Split the table into three parts
        part1_table = l2_norm_table.iloc[:, :4]
        part2_table = l2_norm_table.iloc[:, 4:8]
        part3_table = l2_norm_table.iloc[:, 8:]

        with PdfPages('l2_norm_table_normalized_single_model_homo_unfiltered_new.pdf') as pdf:
            fig, axs = plt.subplots(3, 1, figsize=(11.0, 8.5))
            fig.suptitle('Normalized L2 Norm of LSTM Predictions vs Actual data (Strategy 1, Homogeneous Unfiltered New Data)', fontsize=14)

            for ax, part_table in zip(axs, [part1_table, part2_table, part3_table]):
                ax.axis('tight')
                ax.axis('off')
                part_table_str = part_table.map(lambda x: f"{x:.16f}")
                part_table_plot = ax.table(cellText=part_table_str.values, colLabels=part_table.columns,
                                           rowLabels=part_table.index, cellLoc='center', loc='center')
                part_table_plot.auto_set_font_size(False)
                part_table_plot.set_fontsize(8)
                part_table_plot.scale(1.2, 1.2)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    tabulate_l2_norm()

if __name__ == '__main__':
    main()