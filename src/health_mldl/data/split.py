from sklearn.model_selection import train_test_split


def train_val_test_split(
    x,
    y,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    val_ratio_of_train = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_ratio_of_train,
        random_state=random_state,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
