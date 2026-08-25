# Run storage

Generation runs are immutable runtime artifacts and are ignored by Git. Every generation receives a unique directory containing its request, durable state, logs, provenance, intermediates, and outputs. Artist training output remains under the corresponding private artist root.

Failed runs are retained for diagnosis. Resume and reproduction commands remain backlog work; future reproduction creates a new child rather than overwriting its parent.
