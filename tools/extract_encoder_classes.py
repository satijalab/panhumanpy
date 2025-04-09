import pathlib
import pickle

import click
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.option(
    "--encoders",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the pickle file containing inference encoders.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Directory path where output label files will be saved.",
)
def main(encoders, output):
    """Parse label encoders and output their class labels as text files.

    This function loads a pickle file containing a list of encoder objects. For each encoder,
    it re-fits a new LabelEncoder using the encoder's classes, verifies the consistency of the
    transformations, and writes the class labels to an output file.

    Parameters:
        encoders (str): File path to the pickle file with inference encoders.
        output (str): Directory where the output text files will be saved.
    """

    path_to_encoders = pathlib.Path(encoders)
    path_to_output = pathlib.Path(output)

    # Create the output directory if it does not exist.
    path_to_output.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading encoders from: {path_to_encoders}")
    with open(path_to_encoders, "rb") as fp:
        encoders_obj: list[LabelEncoder] = pickle.load(fp)

    click.echo(f"Loaded {len(encoders_obj)} encoder(s).")
    for i, encoder in enumerate(encoders_obj):
        click.echo(f"Processing encoder {i}...")
        # Retrieve the classes from the encoder.
        encoder_classes: list[str] = encoder.classes_  # type: ignore [reportAssignmentType]
        # Fit a new LabelEncoder using the same classes.
        new_encoder = LabelEncoder().fit(encoder_classes)

        # Assert that the new encoder's classes match the original encoder's classes.
        assert (new_encoder.classes_ == encoder_classes).all()  # type: ignore [reportAttributeAccessIssue]

        # Verify that transforming each class with the encoder remains consistent.
        for class_ in encoder_classes:
            encoding = encoder.transform([class_])
            new_encoding = encoder.transform([class_])
            assert new_encoding == encoding  # Ensure encoding consistency

            # Validate that inverse transformation returns the original class.
            new_class = new_encoder.inverse_transform(new_encoding)
            assert new_class == class_

        # Write the encoder's class labels to the file.
        path_to_labels = path_to_output / f"encoder_classes_level{i}.txt"
        click.echo(f"Writing output to: {path_to_labels}")
        with open(path_to_labels, "w") as file:
            for line in encoder_classes:
                file.write(f"{line}\n")

    click.echo("Done.")


if __name__ == "__main__":
    main()
