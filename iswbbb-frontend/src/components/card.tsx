"use client";

import {Resizable} from 're-resizable';

import React, { useState, useEffect } from 'react'  
import useDragger from "../hooks/useDragger";

type cardProps = {
    url: string
};

const Card: React.FC<cardProps & { onUpdate: (url:string, x: number, y: number, w: number, h: number) => void }> = ({ url, onUpdate }) => {
  
  useDragger(`${url}`);

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

    function handleStop(e: any, dragElement: any) {
        setX(dragElement.x);
        setY(dragElement.y);
    }

    function handleResizeStop(e: any, direction: any, ref: any, d: any) {
        setW(ref.clientWidth);
        setH(ref.clientHeight);
        onUpdate(url, x, y, ref.clientWidth, ref.clientHeight);
    }

  return <div id={url} className="object">

            <Resizable
                defaultSize={{
                    width: 200,
                    height: 200,
                }}
                style={{
                    background: `url(${url})`,
                    backgroundSize: 'contain',
                    backgroundRepeat: 'no-repeat'
                }}
                onResizeStop={handleResizeStop}
                bounds="window"
                >
            </Resizable>
    </div>
};

export default Card;