import { configureStore } from '@reduxjs/toolkit'

import statesSlice from '../pages/stateSlice';

export default configureStore({
    reducer: {
        states: statesSlice
    },
})