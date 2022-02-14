
var mongoose = require('mongoose')
var Schema = mongoose.Schema;

var depressionSchema = new Schema({
    is_depressed: {
        type: Boolean
    },
    user_id: {
        type: mongoose.Types.ObjectId,
        ref: 'User'
    },
    audio_id: {
        type: mongoose.Types.ObjectId,
        ref: 'Audio'
    }
})

module.exports = mongoose.model('Depression', depressionSchema);

