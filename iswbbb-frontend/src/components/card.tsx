"use client";
import React, { useState } from 'react'


import {Resizable} from 're-resizable';
import Draggable from 'react-draggable';

type cardProps = {
    url: string
};

const card:React.FC<cardProps> = ({ url }) => {
    return(
        <Draggable axis="both"
        defaultPosition={{x: 0, y: 0}}
        grid={[25, 25]}
        bounds='parent'
        scale={1}
        >
            <Resizable
                defaultSize={{
                    width: 100,
                    height: 100,
                }}
                style={{
                    background: `url(${url})`,
                    backgroundSize: 'contain',
                    backgroundRepeat: 'no-repeat'
                }}
                >
            </Resizable>
        </Draggable>
        
    )
}
export default card;