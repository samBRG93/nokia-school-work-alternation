from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def plot_loss(loss, plot_name=None, show: bool = True):
    epochs = len(loss['validation_0']['rmse'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, loss['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, loss['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('Regression Mean Squared Error')
    plt.title('Loss')

    if plot_name:
        plt.savefig(plot_name)
    if show:
        plt.show()


def plot_scatter(y_test, y_pred, plot_name=None, show: bool = True):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed")
    plt.xlabel("Expected Value")
    plt.ylabel("Prediction")
    plt.title("Predictions vs Expected Values")
    plt.grid(True)

    if plot_name:
        plt.savefig(plot_name)
    if show:
        plt.show()


def plot_temperature(x_axis, y_test, y_pred, plot_name=None, show: bool = True):
    plt.figure(figsize=(15, 7), dpi=150)
    plt.plot(x_axis, y_pred, label="Prediction", color="blue", marker="o", linestyle="-")
    plt.scatter(x_axis, y_test, label="Expected Values", color="red", marker="x", alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Predicted vs Expected Temperature")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    if plot_name:
        plt.savefig(plot_name)
    if show:
        plt.show()
