from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, FileField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from flask_wtf.file import FileAllowed
from models import User

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class UploadForm(FlaskForm):
    resume = FileField('Resume', validators=[DataRequired(), FileAllowed(['pdf', 'docx'], 'PDF or DOCX only')])
    job_role = StringField('Job Role', validators=[DataRequired()])
    submit = SubmitField('Analyze')

class ProfileForm(FlaskForm):
    is_premium = BooleanField('Premium User')
    submit = SubmitField('Update')

class LinkedInForm(FlaskForm):
    url = StringField('LinkedIn Profile URL', validators=[DataRequired()])
    submit = SubmitField('Import')

class RewriteForm(FlaskForm):
    section = TextAreaField('Section to Rewrite', validators=[DataRequired()])
    job_desc = TextAreaField('Job Description', validators=[DataRequired()])
    submit = SubmitField('Rewrite')

class CoverForm(FlaskForm):
    resume_id = SelectField('Select Resume', coerce=int, validators=[DataRequired()])
    job_id = SelectField('Select Job', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Generate Cover Letter')
