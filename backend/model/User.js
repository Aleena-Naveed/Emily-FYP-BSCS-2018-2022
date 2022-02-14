var mongoose = require('mongoose')
var Schema = mongoose.Schema;
var passportLocalMongoose = require('passport-local-mongoose');

var userSchema = new Schema({
    name: {
        type: String
    },
    email: {
        type: String
    },
    contact: {
        type: String
    },
    contactIsVerified: {
        type: Boolean,
        default: false
    },
    emailIsVerified: {
        type: Boolean,
        default: false
    },
    gender: {
        type: String
    },
    occupation: {
        type: String
    },
    dob: {
        type: String
    },
    age: {
        type: Number
    },
    visit: {
        type: Number,
        default: 0
    },
    country: {
        type: String
    }
})

userSchema.plugin(passportLocalMongoose);

module.exports = mongoose.model('User', userSchema);