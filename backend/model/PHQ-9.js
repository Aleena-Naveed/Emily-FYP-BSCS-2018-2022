
var mongoose = require('mongoose')
var Schema = mongoose.Schema;

var phqSchema = new Schema({
    phq_score: {
        type: Number
    },
    depression_level: {
        type: String
    },
    user_id: {
        type: mongoose.Types.ObjectId,
        ref: 'User'
    }
})

module.exports = mongoose.model('PHQ', phqSchema);

