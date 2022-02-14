
var mongoose = require('mongoose')
var Schema = mongoose.Schema;

var audioSchema = new Schema({
    audio_path: {
        type: String
    },
    user_id: {
        type: mongoose.Types.ObjectId,
        ref: 'User'
    }
})

module.exports = mongoose.model('Audio', audioSchema);

