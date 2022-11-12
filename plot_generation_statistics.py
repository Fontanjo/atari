import numpy as np
import re
import matplotlib.pyplot as plt



FILE_PATH = "./logs/221104_1741.log"

def main():

    # To match lines as 'From the 8460 new images, 809 added to the list in 104.26s',
    correctLineRE = re.compile(r'From the \d* new images, \d* added to the list in \d*.\d*s')

    # Save values
    added = []
    time = []
    with open(FILE_PATH, 'r') as file:
        for line in file:
            # Match lines
            s = correctLineRE.search(line)
            # Extract values
            if s:
                new_img, added_img, added_time = re.search(r'(\b\d+).+(\b\d+).+(\b\d+\.\d+)', s.string).groups()
                added.append(int(added_img))
                time.append(float(added_time))

    # Create cumulative sum
    added_sum = np.cumsum(added)
    time_sum = np.cumsum(time) / 3600

    # Create moving average
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, 'same')
    MA = 10
    added_ma = moving_average(added, MA)


    # Plot
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(time_sum, added_sum, color='Green')
    plt.xlabel('Cumulative time (h)')
    plt.ylabel('Cumulative images')
    plt.title('Total added images')

    # plt.figure()
    plt.subplot(1, 2, 2)
    plt.scatter(time_sum, added)
    plt.xlabel('Cumulative time (h)')
    plt.ylabel('Newly added images')
    plt.plot(time_sum, added_ma,
            label=f'Moving average ({MA})',
            color='red')

    plt.axhline(10,
            label='Stopping condition',
            color='darkred')
    plt.title('Newly added images')

    plt.legend()
    plt.savefig(FILE_PATH[:-3])
    plt.show()




if __name__ == "__main__":
    main()
