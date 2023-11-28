"use client";
import React, { useState, useEffect } from 'react'


import {Resizable} from 're-resizable';
import Draggable from 'react-draggable';

type cardProps = {
    url: string
    x: number
    y: number
};

const card:React.FC<cardProps> = ({ url }) => {
    const [x, setX]= useState(0)
    const [y, setY]= useState(0)
    const [w, setW]= useState(100)
    const [h, setH]= useState(100)

    useEffect(() => {
        console.log('Current position X:', x);
        console.log('Current position Y:', y);
    }, [x, y]);

    useEffect(() => {
        console.log('Current width:', w);
        console.log('Current height:', h);
    }, [w, h]);

    function handleStop(e, dragElement) {
        setX(dragElement.x);
        setY(dragElement.y);
    }

    function handleResizeStop(e, direction, ref, d) {
        setW(ref.clientWidth);
        setH(ref.clientHeight);
    }

    return(
        <Draggable axis="both"
        defaultPosition={{x: 0, y: 0}}
        position={{x: x, y:y}}
        onStop={handleStop} 
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
                onResizeStop={handleResizeStop}
                >
            </Resizable>
        </Draggable>
        
    )
}
export default card;