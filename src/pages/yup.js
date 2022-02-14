import * as yup from "yup";
import "yup-phone";

export const basicVal = yup.object({
    username: yup
        .string('Enter Your First Name')
        .required('First Name is Required'),
    email: yup
        .string('Enter your email')
        .email('Enter a valid email')
        .required('Email is required'),
    password: yup
        .string('Enter your password')
        .min(8, 'Password should be of minimum 8 characters length')
        .required('Password is Required'),
    confirmpassword: yup
        .string("Confirm Password")
        .required("Confirm Password is Required")
        .test('passwords-match', 'Passwords must match', function (value) {
            return this.parent.password === value
        }),
    phone_number: yup
        .string("Enter Phone Number")
        .required("Phone Number is Required")
        .phone()
})

export const validatePin = yup.object({
    smspin: yup
        .string()
        .matches(/^[0-9]*$/)
        .required('SMS PIN is Required'),
    emailpin: yup
        .string()
        .matches(/^[0-9]*$/)
        .required('Email PIN is Required'),
})

export const userprofile = yup.object({
    fullname: yup
        .string('Enter Full Name')
        .required('Full Name is Required'),
    country: yup
        .string('Enter Country')
        .required('Country is Required'),
    occupation: yup
        .string('Enter Occupation')
        .required('Occupation is Required'),
    gender: yup
        .string('Enter Gender')
        .required('Gender is Required'),
    dob: yup
        .date("Enter Date of Birth")
        .required("Date of Birth is Required")
        .test("age", "You must be 18 or older", function (birthdate) {
            const cutoff = new Date();
            cutoff.setFullYear(cutoff.getFullYear() - 18);
            return birthdate <= cutoff;
        }),

})