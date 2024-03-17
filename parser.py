import json
import uuid
import os
import os.path as osp


if __name__ == "__main__":

    # path to input images
    img_input_path = "/Users/julianludt/Desktop/datasets/original/ModelAgencyData/cleaned_model_data.json"
    ratings_to_image = "/Users/julianludt/Desktop/datasets/original/ModelAgencyData/attributes.json"

    attributes_male = "average, big, broad_shoulders, delicate_build, long_legs, long_neck, long_torso, masculine, muscular, rectangular, short, short_arms, skinny_arms, soft_body, tall, weight, height, age"
    attributes_female = "big, broad_shoulders, large_breasts, long_legs, long_neck, long_torso, muscular, pear_shaped, petite, short, short_arms, skinny_legs, slim_waist, tall, feminine, weight, height, age"

    # Initialize list to hold all JSON data
    json_data_list = []

    data = json.load(open(img_input_path, 'r'))
    ratings = json.load(open(ratings_to_image, 'r'))

    for agency, val in data.items():
        for model, urls, gender in zip(val['model_name'], val['image_urls'], val['gender']):
            for url in urls:
                _, fname = osp.split(osp.abspath(url))
                fname = fname.split("?", 1)[0]
                img_name = os.path.join(agency, 'images', model, fname)
                # Create a unique ID for each image
                unique_id = str(uuid.uuid4())

                # Structure for LLaVA JSON
                if model in ratings[agency]:
                    rating = ratings[agency][model]['attributes']
                    json_data = {
                        "id": unique_id,
                        "image": img_name,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"Detect the gender of the person on the photo as male or female. If the detected gender is male rate how much each of the following words applies to the body shape: {attributes_male}. If the detected gender is female rate how much each of the following words applies to the body shape: {attributes_female}. The rating values can range from 1 (does not apply) to 5 (does apply). The expected output format is: 'gender: rating_values'."
                            },
                            {
                                "from": "gpt",
                                "value": gender + ": " + str(rating)[1:-1] + ", " + str(ratings[agency][model]['guess_weight']) + ", " + str(ratings[agency][model]['guess_height']) + ", " + str(ratings[agency][model]['guess_age'])
                            }
                        ]
                    }
                    # Append to list
                    json_data_list.append(json_data)

    #Save the JSON data list to a file
    json_output_path = os.path.join(".", 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


