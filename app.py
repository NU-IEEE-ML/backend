import flask
from flask import Flask, request
import xgboost
import pandas as pd
    
app = flask.Flask(__name__, template_folder='templates')

@app.route('/api/form', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        print("connection head")
        content = request.get_json()
        print(content)
        
        bst = xgboost.Booster({'nthread' : 4})
        
        location = content['Location']
        if location == 'Chicago':
            bst.load_model('chicago_model.bin')
        elif location == 'New York':
            bst.load_model('newyork_model.bin')
        elif location == 'Los Angeles':
            bst.load_model('la_model.bin')
            
        all_property_types = ['property_type_Apartment','property_type_Hotel','property_type_House',
                              'property_type_Other']
        all_room_types = ['room_type_Enter home/apt','room_type_Hotel room','room_type_Private room',
                          'room_type_Shared room']
        all_amenities = ['bbq','balcony','bed_linen','breakfast','tv','coffee_machine',
                         'cooking_basics','white_goods','elevator','gym','parking','outdoor_space',
                         'hot_tub_sauna_or_pool','internet','long_term_stays','private_entrance']
        all_response_rates = ['host_response_rate_0-49%','host_response_rate_50-89%','host_response_rate_90-99%',
                              'host_response_rate_100%']
        
        all_review_scores = ['review_scores_rating_0-49/100',
       'review_scores_rating_50-79/100', 'review_scores_rating_80-94/100',
       'review_scores_rating_95-100/100']
        
        all_accuracy_scores = ['review_scores_accuracy_0-5/10', 'review_scores_accuracy_10/10',
       'review_scores_accuracy_8/10', 'review_scores_accuracy_9/10']
        all_clean_scores = ['review_scores_cleanliness_0-5/10',
       'review_scores_cleanliness_10/10', 'review_scores_cleanliness_8/10',
       'review_scores_cleanliness_9/10']
        all_checkin_scores = ['review_scores_checkin_0-5/10', 'review_scores_checkin_10/10',
       'review_scores_checkin_8/10', 'review_scores_checkin_9/10']
        all_comm_scores = ['review_scores_communication_0-5/10',
       'review_scores_communication_10/10', 'review_scores_communication_8/10',
       'review_scores_communication_9/10']
        all_location_scores = ['review_scores_location_0-5/10', 'review_scores_location_10/10',
       'review_scores_location_8/10', 'review_scores_location_9/10']
        all_value_scores = ['review_scores_value_0-5/10',
       'review_scores_value_10/10', 'review_scores_value_8/10',
       'review_scores_value_9/10']
        all_scores = all_accuracy_scores+all_clean_scores+all_checkin_scores+all_comm_scores+all_location_scores+all_value_scores

        all_first_reviews = ['time_since_first_review_0-6 months',
       'time_since_first_review_1-2 years',
       'time_since_first_review_2-3 years', 'time_since_first_review_4+ years',
       'time_since_first_review_6-12 months']
        all_last_reviews = ['time_since_last_review_0-2 weeks', 'time_since_last_review_1+ year',
       'time_since_last_review_2-6 months', 'time_since_last_review_2-8 weeks',
       'time_since_last_review_6-12 months']
        
        features = ['host_is_superhost','host_listings_count','host_identity_verified',
                    'accommodates','bedrooms','beds',
                    'minimum_nights','maximum_nights',
                    'availability_90',
                    'number_of_reviews']
        
        # List of features that will be used
        chosen_features = []
        for i in range(len(features)):
            chosen_features.append(content[features[i]])
            
        features += all_amenities + all_response_rates + all_property_types + all_room_types + all_review_scores + all_scores
        
        property_type = content['property_type_']
        room_type = content['room_type_']
        amenities = content['Amenities']
        
        response_rate = content['host_response_rate_']
        review_rate = content['review_scores_rating']
        
        acc_rate = content['review_accuracy_rating']
        clean_rate = content['review_cleanliness_rating']
        check_rate = content['review_checkin_rating']
        comm_rate = content['review_communication_rating']
        loc_rate = content['review_location_rating']
        value_rate = content['review_value_rating']
        
        scores = [acc_rate,clean_rate,check_rate,comm_rate,loc_rate,value_rate]

        # Sanitizing specific feature types
        all_property_types = [0 if i != property_type else 1 for i in all_property_types]
        
        all_room_types = [0 if i != room_type else 1 for i in all_room_types]
        
        for i in amenities:
            if i in all_amenities:
                all_amenities[all_amenities.index(i)] = 1     
        all_amenities = [0 if i != 1 else i for i in all_amenities]
        
        if response_rate >= 0 and response_rate <= 50:
            all_response_rates[all_response_rates.index('host_response_rate_0-49%')] = 1
        if response_rate >= 50 and response_rate <= 89:
            all_response_rates[all_response_rates.index('host_response_rate_50-89%')] = 1
        if response_rate >= 90 and response_rate <= 99:
            all_response_rates[all_response_rates.index('host_response_rate_90-99%')] = 1
        if response_rate == 100:
            all_response_rates[all_response_rates.index('host_response_rate_100%')] = 1
        all_response_rates = [0 if i != 1 else i for i in all_response_rates]
        
        if review_rate >= 0 and review_rate < 50:
            all_review_scores[all_review_scores.index('review_scores_rating_0-49/100')] = 1
        if review_rate >= 50 and review_rate <= 79:
            all_review_scores[all_review_scores.index('review_scores_rating_50-79/100')] = 1
        if review_rate >= 80 and review_rate <= 94:
            all_review_scores[all_review_scores.index('review_scores_rating_80-94/100')] = 1
        if review_rate >= 95 and review_rate <= 100:
            all_review_scores[all_review_scores.index('review_scores_rating_95-100/100')] = 1
        all_review_scores = [0 if i != 1 else i for i in all_review_scores]
        
        for i in range(len(scores)):
            if scores[i] >= 0 and scores[i] <= 5:
                    all_scores[i*4] = 1
            if scores[i] == 10:
                    all_scores[i*4 + 1] = 1
            if scores[i] == 8:
                    all_scores[i*4 + 2] = 1
            if scores[i] == 9:
                    all_scores[i*4 + 3] = 1
        all_scores = [0 if i != 1 else i for i in all_scores]
                
        chosen_features += all_amenities + all_property_types + all_room_types + all_response_rates + all_review_scores + all_scores
        
        input_variables = pd.DataFrame([chosen_features],
                                       columns=features,
                                       dtype=float)
        d_input = xgboost.DMatrix(input_variables)
        prediction = bst.predict(d_input)[0]
        return str(prediction)
        '''
        bedrooms = flask.request.form['bedrooms']
        accom = flask.request.form['accommodations']
        
        input_variables = pd.DataFrame([[bedrooms, accom]],
                                       columns=['bedrooms', 'accommodations'],
                                       dtype=float)
        d_input = xgboost.DMatrix(input_variables)
        prediction = bst.predict(d_input)[0]
        print("prediction = ", prediction)
        return prediction
        '''
    
    '''
    flask.render_template('main.html', original_input={'Bedrooms':bedrooms,
                                                                  'Accomadates':accom,
                                                                  'Gym':gym},
                                     result=prediction,)
    
    '''

if __name__ == '__main__':
    app.run()
