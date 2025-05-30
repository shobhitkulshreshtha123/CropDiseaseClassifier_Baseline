import numpy as np

def apple_p(feature_vector, model):
    processed_vector = np.array(feature_vector).reshape(1, -1)
    output = model.predict(processed_vector)
    probabilities = model.predict_proba(processed_vector)

    output = int(output)
    label_dict = {
        0: 'Healthy',
        1: 'Apple scab',
        2: 'Black rot',
        3: 'Cedar apple rust'
    }

    confidence = round(100 * np.max(probabilities), 2)  # in %
    predicted_label = label_dict[output]
    return predicted_label, confidence

def corn_p(feature_vector,model):
    processed_vector = np.array(feature_vector).reshape(1, -1)
    output = model.predict(processed_vector)
    probabilities = model.predict_proba(processed_vector)

    output = int(output)
    label_dict = {
		0: 'Healthy',
		1: 'Cercospora leaf spot (Gray leaf spot)',
		2: 'Common rust',
		3: 'Northern Leaf Blight'
    }

    confidence = round(100 * np.max(probabilities), 2)  # in %
    predicted_label = label_dict[output]
    return predicted_label, confidence

def grapes_p(feature_vector,model):
    processed_vector = np.array(feature_vector).reshape(1, -1)
    output = model.predict(processed_vector)
    probabilities = model.predict_proba(processed_vector)

    output = int(output)
    label_dict = {
		0 : 'Healthy',
		1 : 'Black rot',
		2 : 'Esca (Black Measles)',
		3 : 'Leaf blight (Isariopsis Leaf Spot)'
    }

    confidence = round(100 * np.max(probabilities), 2)  # in %
    predicted_label = label_dict[output]
    return predicted_label, confidence

def potato_p(feature_vector,model):
    processed_vector = np.array(feature_vector).reshape(1, -1)
    output = model.predict(processed_vector)
    probabilities = model.predict_proba(processed_vector)

    output = int(output)
    label_dict = {
		0: 'Healthy',
		1: 'Early blight',
		2: 'Late blight'
    }

    confidence = round(100 * np.max(probabilities), 2)  # in %
    predicted_label = label_dict[output]
    return predicted_label, confidence

def tomato_p(feature_vector,model):
    processed_vector = np.array(feature_vector).reshape(1, -1)
    output = model.predict(processed_vector)
    probabilities = model.predict_proba(processed_vector)

    output = int(output)
    label_dict = {
		0 : 'Healthy',
		1 : 'Bacterial spot',
		2 : 'Early blight',
		3 : 'Late blight',
		4 : 'Leaf Mold',
		5 : 'Septoria leaf spot',
		6 : 'Spider mites (Two-spotted spider mite)',
		7 : 'Target Spot',
		8 : 'Yellow Leaf Curl Virus',
		9 : 'Mosaic virus'
    }

    confidence = round(100 * np.max(probabilities), 2)  # in %
    predicted_label = label_dict[output]
    return predicted_label, confidence
