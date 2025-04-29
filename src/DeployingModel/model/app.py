import gradio as gr
import joblib
import numpy as np

def weighted_logloss(y_true, y_pred):
    preds = 1.0 / (1.0 + np.exp(-y_pred))  # Dự đoán thô sigmoid
    weight_pos = 2
    weight_neg = 1
    grad = np.where(y_true == 1, weight_pos * (preds - 1), weight_neg * preds)
    hess = np.where(y_true == 1, weight_pos * preds * (1 - preds), weight_neg * preds * (1 - preds))
    return grad, hess

model= joblib.load('./model.pkl')

# input features 
features = [
    'Age',
    'Academic Pressure',
    'Work Pressure',
    'CGPA',
    'Study Satisfaction',
    'Job Satisfaction',
    'Suicidal Thoughts',
    'Work/Study Hours',
    'Financial Stress',
    'Mental Illness History',
    'Gender_male',
    "City_'less than 5 kalyan'",
    'City_3.0',
    'City_agra',
    'City_ahmedabad',
    'City_bangalore',
    'City_bhavna',
    'City_bhopal',
    'City_chennai',
    'City_city',
    'City_delhi',
    'City_faridabad',
    'City_gaurav',
    'City_ghaziabad',
    'City_harsh',
    'City_harsha',
    'City_hyderabad',
    'City_indore',
    'City_jaipur',
    'City_kalyan',
    'City_kanpur',
    'City_khaziabad',
    'City_kibara',
    'City_kolkata',
    'City_lucknow',
    'City_ludhiana',
    'City_m.com',
    'City_m.tech',
    'City_me',
    'City_meerut',
    'City_mihir',
    'City_mira',
    'City_mumbai',
    'City_nagpur',
    'City_nalini',
    'City_nalyan',
    'City_nandini',
    'City_nashik',
    'City_patna',
    'City_pune',
    'City_rajkot',
    'City_rashi',
    'City_reyansh',
    'City_saanvi',
    'City_srinagar',
    'City_surat',
    'City_thane',
    'City_vaanya',
    'City_vadodara',
    'City_varanasi',
    'City_vasai-virar',
    'City_visakhapatnam',
    "Profession_content writer",
    "Profession_digital marketer",
    "Profession_educational consultant",
    "Profession_ux/ui designer",
    'Profession_architect',
    'Profession_chef',
    'Profession_doctor',
    'Profession_entrepreneur',
    'Profession_lawyer',
    'Profession_manager',
    'Profession_pharmacist',
    'Profession_student',
    'Profession_teacher',
    'Degree_b.arch',
    'Degree_b.com',
    'Degree_b.ed',
    'Degree_b.pharm',
    'Degree_b.tech',
    'Degree_ba',
    'Degree_bba',
    'Degree_bca',
    'Degree_be',
    'Degree_bhm',
    'Degree_bsc',
    'Degree_llb',
    'Degree_llm',
    'Degree_m.com',
    'Degree_m.ed',
    'Degree_m.pharm',
    'Degree_m.tech',
    'Degree_ma',
    'Degree_mba',
    'Degree_mbbs',
    'Degree_mca',
    'Degree_md',
    'Degree_me',
    'Degree_mhm',
    'Degree_msc',
    'Degree_others',
    'Degree_phd',
    'Dietary Habits_moderate',
    'Dietary Habits_others',
    'Dietary Habits_unhealthy',
    'Sleep Duration_7-8 hours',
    'Sleep Duration_less than 5 hours',
    'Sleep Duration_more than 8 hours',
    'Sleep Duration_others'
]


def process_input(input_data):
    keys=['gender','age','city','profession','academic_pressure','work_pressure','cgpa','study_satisfaction',
    'job_satisfaction','sleep_duration','dietary_habits','degree','suicidal_thoughts','work_study_hours','financial_stress','mental_illness_history']

    input_dict = dict(zip(keys, input_data))
    input_processed = []

    data={
        'Age': input_dict['age'],
        'Academic Pressure': input_dict['academic_pressure'],
        'Work Pressure': input_dict['work_pressure'],
        'CGPA': input_dict['cgpa'],
        'Study Satisfaction': input_dict['study_satisfaction'],
        'Job Satisfaction': input_dict['job_satisfaction'],
        'Suicidal Thoughts': 1 if str(input_dict['suicidal_thoughts']).lower() == 'yes' else 0,
        'Work/Study Hours': input_dict['work_study_hours'],
        'Financial Stress': input_dict['financial_stress'],
        'Mental Illness History': 1 if str(input_dict['mental_illness_history']).lower() == 'yes' else 0,
        'Gender_male': 1 if str(input_dict['gender']).lower() == 'male' else 0,
    }

    one_hot_columns = [
        f"City_{input_dict['city'].lower()}",
        f"Profession_{input_dict['profession'].lower()}",
        f"Degree_{input_dict['degree'].lower()}",
        f"Dietary Habits_{input_dict['dietary_habits'].lower()}",
        f"Sleep Duration_{input_dict['sleep_duration'].lower()}",
    ]

    input_vector = {}
    for feature in features:
        if feature in data:
            input_vector[feature] = data[feature]
        elif feature in one_hot_columns:
            input_vector[feature] = 1
        else:
            input_vector[feature] = 0  # Mặc định 0 cho các feature không liên quan
    
    input_processed = list(input_vector.values())
    
    return input_processed

def predict_depression(gender, age, city, profession, academic_pressure, work_pressure, cgpa, study_satisfaction,job_satisfaction, 
                       sleep_duration, dietary_habits, degree, suicidal_thoughts, work_study_hours, financial_stress, mental_illness_history):
    try:
        input_data = [gender, age, city, profession, academic_pressure, work_pressure, cgpa, study_satisfaction,job_satisfaction, 
                        sleep_duration, dietary_habits, degree, suicidal_thoughts, work_study_hours, financial_stress, mental_illness_history]

        
        input_processed = process_input(input_data)

        prediction= model.predict([input_processed])[0]

        return 'Depressed.' if prediction == 1 else 'Not Depressed'
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", font=[gr.themes.GoogleFont("Roboto")]),
            css="""

            .block{
                background-color: rgba(255, 255, 255, 0.8); /* Nền trắng bán trong suốt cho Accordion */
            }

            .svelte-1w6vloh { /*thay đổi cho thẻ label*/
                font-weight: 600; /* Đậm chữ */
                font-size:120%;
                border-radius: 10px;    /* Bo góc cho thẻ chứa */
                background-color: rgba(255, 255, 255, 0.8);  /* Nền trắng bán trong suốt cho Accordion */
                padding:10px;
            }


            """
) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🧠 Depression Prediction App</h1>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<h3 style='text-align: center;'>Please fill in the following details:</h3>")
            with gr.Accordion("🧍 Personal Information", open=True):
                gender = gr.Dropdown(['male', 'female'], label="Gender")

                age = gr.Number(minimum=0, maximum=50,label="Age")

                city=gr.Dropdown(['Agra','Ahmedabad','Bangalore','Bhavna','Bhopal','Chennai','City','Delhi','Faridabad',
                        'Gaurav','Ghaziabad','Harsh','Harsha','Hyderabad','Indore','Jaipur','Kalyan','Kanpur',
                        'Khaziabad','Kibara','Kolkata','Lucknow','Ludhiana','M.com','M.tech','Me','Meerut','Mihir',
                        'Mira','Mumbai','Nagpur','Nalini','Nalyan','Nandini','Nashik','Patna','Pune','Rajkot','Rashi',
                        'Reyansh','Saanvi','Srinagar','Surat','Thane','Vaanya','Vadodara','Varanasi','Vasai-virar','Visakhapatnam'], label="City")
        
                profession=gr.Dropdown([ 'Architect', 'Chef', "'Civil Engineer'", "'Content Writer'", "'Digital Marketer'", 'Doctor', 
                                "'Educational Consultant'", 'Entrepreneur', 'Lawyer', 'Manager', 'Pharmacist', "'UX/UI Designer'", 'Student', 'Teacher'], label="Profession")
                
                degree=gr.Dropdown(['B.ed','B.pharm','B.tech','Ba','Bba','Bca','Be','Bhm','Bsc','Llb',
                            'Llm','M.com','M.ed','M.pharm','M.tech','Ma','Mba','Mbbs','Mca','Md','Me','Mhm','Msc','Others','Phd'], label="Degree")
        
            with gr.Accordion("📚 Academic & Work Pressure", open=False):
                academic_pressure = gr.Slider(minimum=0, maximum=5, step=1, label="Academic Pressure")

                work_pressure = gr.Slider(minimum=0, maximum=5, step=1, label="Work Pressure")

                cgpa=gr.Number(minimum=0, maximum=10,label="CGPA")

                study_satisfaction = gr.Slider(minimum=0, maximum=5, step=1, label="Study Satisfaction")

                job_satisfaction = gr.Slider(minimum=0, maximum=5, step=1, label="Job Satisfaction")

                work_study_hours = gr.Number(minimum=0, maximum=24,label="Work/Study Hours per day")

                financial_stress = gr.Slider(minimum=0, maximum=5, step=1, label="Financial Stress")

            with gr.Accordion("🛌 Lifestyle Habits", open=False):

                sleep_duration = gr.Dropdown([ 'less than 5 hours','5-6 hours', '7-8 hours','more than 8 hours', 'others'], label="Sleep Duration per day")

                dietary_habits = gr.Dropdown(['unhealthy', 'moderate', 'healthy', 'others'], label="Dietary Habits")

            with gr.Accordion("🧠 Psychological Factors", open=False):

                suicidal_thoughts = gr.Dropdown(['yes', 'no'], label="Suicidal Thoughts")

                mental_illness_history = gr.Dropdown(['yes', 'no'], label="Mental Illness History")
    
    with gr.Row():
        btn = gr.Button("🔮 Predict", variant="primary")

    with gr.Row():
        output = gr.Textbox(label="Prediction Result", lines=2)

    btn.click(predict_depression,
              inputs=[gender, age, city, profession, academic_pressure, work_pressure, cgpa, study_satisfaction,
                       job_satisfaction, sleep_duration, dietary_habits, degree, suicidal_thoughts, work_study_hours, financial_stress, mental_illness_history],
                outputs=output,
    )

    gr.Markdown("<h3 style='text-align: center;'>Developed by: L</h3>")

demo.launch()


