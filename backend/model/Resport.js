var mongoose = require('mongoose')
var Schema = mongoose.Schema;

var reportSchema = new Schema({
    title: {
        type: String
    },
    date: {
        type: String
    },
    link: {
        type: String
    },
    user_id: {
        type: mongoose.Types.ObjectId,
        ref: 'User'
    }
})

module.exports = mongoose.model('Report', reportSchema);

